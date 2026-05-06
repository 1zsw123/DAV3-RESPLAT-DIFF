"""
Dataset loader for GoPro defocus blur (multi-crop "multi-view" supervision).

Dataset structure:
  root/
  ├── blur_crops/<sid>_s<NNN>.png    blurry crops (defocus), 768×768
  ├── sharp_crops/<sid>_s<NNN>.png   sharp GT crops (same name)
  └── poses_dav3_predpose_blur_crops.json   per-crop c2w + (fx,fy,cx,cy,w,h)

Each scene id (sid) has 6 crops at a 2x3 spatial grid over the same source
image. The 6 crops share the same camera extrinsics (c2w) but each has its
own intrinsics — fx,fy uniform; cx,cy shifted to reflect the crop origin.
We treat the 6 crops as a multi-view bundle: 6 ctx (blur) → 6 tgt (sharp),
one-to-one supervision.

Yields:
  - context: N blurry crops with GT pose (shared extrinsics, per-view intrinsics)
  - target:  N sharp crops at the same poses (deblur supervision)
"""

from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class GoProDefocusCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]

    near: float = 0.1
    far: float = 100.0

    baseline_min: float = 0.0
    baseline_max: float = 1000.0
    max_fov: float = 180.0
    make_baseline_1: bool = False
    augment: bool = False
    relative_pose: bool = False
    skip_bad_shape: bool = True

    # GoPro-defocus has 501 image pairs named 100.png ... 600.png.
    # We split deterministically by ID range (last 50 → val, rest → train).
    val_ids_start: int = 551   # ids >= this are validation
    samples_per_scene: int = 1
    val_samples_per_scene: int = 1


@dataclass
class DatasetGoProDefocusCfgWrapper:
    gopro_defocus: GoProDefocusCfg


class DatasetGoProDefocus(IterableDataset):
    cfg: GoProDefocusCfg
    stage: Stage
    view_sampler: ViewSampler  # unused, kept for interface compatibility

    def __init__(self, cfg: GoProDefocusCfg, stage: Stage, view_sampler: ViewSampler) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # {scene_id: [(blur_crop_path, sharp_crop_path), ...]} sorted by crop idx
        self._scenes: dict[str, list[tuple[Path, Path]]] = self._discover_scenes()
        # {filename: pose_entry} — DA3-predicted c2w + per-crop intrinsics
        self._poses: dict[str, dict] = self._load_poses()
        # depths_dav3_planB/sharp/{sid}.npy native depth maps (lazy-loaded)
        self._sharp_depth_dir: Path | None = None
        for root in self.cfg.roots:
            d = Path(root) / "depths_dav3_planB" / "sharp"
            if d.exists():
                self._sharp_depth_dir = d
                break
        self._depth_cache: dict[str, np.ndarray] = {}

    def _load_poses(self) -> dict[str, dict]:
        """Load Plan B v3 pose JSON. Each entry has identity c2w (single-view DA3
        inference per crop) + normalized intrinsics already scaled to crop 768x768."""
        out: dict[str, dict] = {}
        for root in self.cfg.roots:
            json_path = Path(root) / "poses_dav3_planB_blur.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                entries = json.load(f)
            for e in entries:
                out[e["filename"]] = e
        return out

    # ------------------------------------------------------------------
    def _discover_scenes(self) -> dict[str, list[tuple[Path, Path]]]:
        scenes: dict[str, list[tuple[int, Path, Path]]] = {}
        for root in self.cfg.roots:
            root = Path(root)
            blur_dir = root / "blur_crops"
            sharp_dir = root / "sharp_crops"
            if not (blur_dir.exists() and sharp_dir.exists()):
                continue
            try:
                files = sorted(os.listdir(blur_dir))
            except OSError:
                continue
            for f in files:
                if not f.endswith(".png"):
                    continue
                stem = f[:-4]              # e.g. "100_s003"
                if "_s" not in stem:
                    continue
                sid, sidx = stem.rsplit("_s", 1)
                if not sid.isdigit() or not sidx.isdigit():
                    continue
                sharp_path = sharp_dir / f
                if not sharp_path.exists():
                    continue
                fid = int(sid)
                is_val = fid >= self.cfg.val_ids_start
                if (self.stage == "train" and not is_val) or (
                    self.stage != "train" and is_val
                ):
                    scenes.setdefault(sid, []).append((int(sidx), blur_dir / f, sharp_path))
        # sort crops within each scene by index, drop the index now that order is set
        return {sid: [(b, s) for _, b, s in sorted(items, key=lambda t: t[0])]
                for sid, items in scenes.items()}

    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> Float[Tensor, "3 h w"]:
        img = Image.open(path).convert("RGB")
        h_tgt, w_tgt = self.cfg.input_image_shape
        orig_w, orig_h = img.size
        target_ar = w_tgt / h_tgt
        src_ar = orig_w / orig_h
        if src_ar > target_ar:          # too wide → crop sides
            new_w = int(orig_h * target_ar)
            left = (orig_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, orig_h))
        elif src_ar < target_ar:        # too tall → crop top/bottom
            new_h = int(orig_w / target_ar)
            top = (orig_h - new_h) // 2
            img = img.crop((0, top, orig_w, top + new_h))
        img = img.resize((w_tgt, h_tgt), Image.LANCZOS)
        return self.to_tensor(img)

    def _depth_for(self, blur_path: Path) -> Tensor | None:
        """Load DA3-on-sharp depth (native 1120x1680), crop to 768x768 at the
        crop_origin from the pose JSON, then resize to input_image_shape.
        Returns [H, W] tensor in metres (DA3 absolute scale), or None if missing."""
        if self._sharp_depth_dir is None:
            return None
        entry = self._poses.get(blur_path.name)
        if entry is None:
            return None
        sid = str(entry["scene_id"])
        if sid not in self._depth_cache:
            npy_path = self._sharp_depth_dir / f"{sid}.npy"
            if not npy_path.exists():
                return None
            self._depth_cache[sid] = np.load(str(npy_path))
        d_native = self._depth_cache[sid]                 # [H_n, W_n]
        H_n, W_n = d_native.shape
        x0 = int(entry["crop_origin_xy"][0])
        y0 = int(entry["crop_origin_xy"][1])
        cw = int(entry["w_native"])
        ch = int(entry["h_native"])
        # Bounded slice + zero-pad if crop extends past native bounds.
        ys, ye = max(0, y0), min(H_n, y0 + ch)
        xs, xe = max(0, x0), min(W_n, x0 + cw)
        if ye <= ys or xe <= xs:
            return None
        sub = d_native[ys:ye, xs:xe]
        out = np.zeros((ch, cw), dtype=np.float32)
        out[ys - y0: ys - y0 + (ye - ys), xs - x0: xs - x0 + (xe - xs)] = sub
        # Resize to network input.
        H_t, W_t = self.cfg.input_image_shape
        d_t = torch.from_numpy(out).float()[None, None]   # [1, 1, ch, cw]
        d_t = F.interpolate(d_t, size=(H_t, W_t), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        return d_t                                         # [H_t, W_t]

    def _pose_for(self, blur_path: Path) -> tuple[Tensor, Tensor] | None:
        """Plan B v3: c2w is near-identity (per-crop single-view DA3 inference).
        JSON stores c2w as 3x4; pad to 4x4 here.
        fx_norm/cx_norm are already normalized to crop 768x768 → use directly.
        cx_norm can be < 0 or > 1 (off-center crop, principal point outside frame)."""
        entry = self._poses.get(blur_path.name)
        if entry is None:
            return None
        c2w_3x4 = torch.tensor(entry["c2w"], dtype=torch.float32)    # [3,4]
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :] = c2w_3x4
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = float(entry["fx_norm"])
        K[1, 1] = float(entry["fy_norm"])
        K[0, 2] = float(entry["cx_norm"])
        K[1, 2] = float(entry["cy_norm"])
        return c2w, K

    def get_bound(self, bound: Literal["near", "far"], n: int) -> Float[Tensor, "v"]:
        return repeat(torch.tensor(getattr(self.cfg, bound), dtype=torch.float32), "-> v", v=n)

    # ------------------------------------------------------------------
    def __iter__(self):
        scene_items = list(self._scenes.items())
        if self.stage == "train":
            random.shuffle(scene_items)

        # ReSplat encoder needs ≥2 views. Each scene has multiple crops;
        # we sample n_ctx distinct crops per scene as multi-view input.
        n_ctx = max(2, int(getattr(self.cfg.view_sampler, "num_context_views", 2)))
        n_samples = (
            max(1, self.cfg.val_samples_per_scene) if self.stage != "train"
            else max(1, self.cfg.samples_per_scene)
        )

        for scene_id, crops in scene_items:
            n_avail = len(crops)
            if n_avail == 0:
                continue

            # Validation/test: deterministic crop choice per scene.
            if self.stage != "train":
                _rng_state = torch.get_rng_state()
                seed = abs(hash(scene_id)) % (2 ** 31)
                random.seed(seed)
                torch.manual_seed(seed)

            try:
                for _ in range(n_samples):
                    if n_avail >= n_ctx:
                        chosen = random.sample(range(n_avail), n_ctx)
                    else:
                        chosen = [random.randrange(n_avail) for _ in range(n_ctx)]

                    try:
                        blur_imgs = torch.stack([
                            self._load_image(crops[i][0]) for i in chosen
                        ])  # [V, 3, H, W]
                        sharp_imgs = torch.stack([
                            self._load_image(crops[i][1]) for i in chosen
                        ])  # [V, 3, H, W]
                    except (FileNotFoundError, OSError):
                        continue

                    # Build per-view extrinsics + intrinsics from preprocessed pose JSON.
                    # All 6 crops of a scene share c2w; intrinsics differ (cx/cy shifted).
                    poses = [self._pose_for(crops[i][0]) for i in chosen]
                    if any(p is None for p in poses):
                        continue
                    ext_ctx  = torch.stack([p[0] for p in poses])   # [V, 4, 4]
                    intr_ctx = torch.stack([p[1] for p in poses])   # [V, 3, 3]

                    idx_t = torch.tensor(chosen, dtype=torch.long)

                    # GoPro Defocus is paired data: every blur ctx frame has its
                    # own sharp GT at the same view → one-to-one supervision.
                    # n_tgt = n_ctx (no subsampling). model_wrapper resolves the
                    # ctx→tgt mapping via index, so identity mapping works.
                    n_tgt = n_ctx

                    ctx_dict = {
                        "extrinsics":  ext_ctx,
                        "intrinsics":  intr_ctx,
                        "image":       blur_imgs,
                        "sharp_image": sharp_imgs,
                        "near":  self.get_bound("near", n_ctx),
                        "far":   self.get_bound("far",  n_ctx),
                        "index": idx_t,
                        "overlap": torch.ones(n_ctx, dtype=torch.float32),
                    }
                    tgt_dict = {
                        "extrinsics": ext_ctx.clone(),
                        "intrinsics": intr_ctx.clone(),
                        "image":      sharp_imgs,
                        "near":  self.get_bound("near", n_tgt),
                        "far":   self.get_bound("far",  n_tgt),
                        "index": idx_t.clone(),
                    }

                    # GT depth for L_depth supervision (DA3-on-sharp, planB v3).
                    depths = [self._depth_for(crops[i][0]) for i in chosen]
                    if all(d is not None for d in depths):
                        tgt_dict["depth"] = torch.stack(depths)        # [V, H, W]

                    yield {
                        "context": ctx_dict,
                        "target":  tgt_dict,
                        "scene":   scene_id,
                    }
            finally:
                if self.stage != "train":
                    torch.set_rng_state(_rng_state)

    @property
    def data_stage(self) -> Stage:
        if self.stage == "val":
            return "test"
        return self.stage
