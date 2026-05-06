"""
Dataset loader for ScanNet++ blur/sharp pairs.

Dataset structure (prepared by scripts/prepare_scannetpp_v2.py):
  root/
  └── {scene_id}/
      ├── iphone_frames/      frame_XXXXXX.jpg  (blurry iPhone input)
      ├── dslr_sharp_frames/  frame_XXXXXX.jpg  (undistorted DSLR, colour-corrected)
      ├── poses_iphone.json   [{frame, fid, c2w, fx, fy, cx, cy, w, h}, ...]
      └── poses_dslr_sharp.json  [{fid, dslr_frame, c2w, fx, fy, cx, cy, w, h,
                                   dslr_dist, dslr_angle}, ...]

Sampling protocol (1-to-1 paired):
  Sample n_views matched pairs from poses_dslr_sharp.json.
  context[i] = blurry iPhone frame for pair i  (iPhone c2w)
  target[i]  = sharp DSLR frame for pair i     (DSLR c2w, ≤0.2m & ≤15° from iPhone)
  Every context frame has exactly one corresponding sharp target frame.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class ScannetppBlurCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]

    near: float
    far: float

    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool

    scene_list: list[str] = field(default_factory=list)
    samples_per_scene: int = 100
    sharp_context: bool = False
    num_context_views: int = 6
    min_sharp_frames: int = 36   # skip scenes with fewer valid DSLR sharp frames


@dataclass
class DatasetScannetppBlurCfgWrapper:
    scannetpp_blur: ScannetppBlurCfg


class DatasetScannetppBlur(IterableDataset):
    cfg: ScannetppBlurCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.01
    far: float = 10.0

    def __init__(self, cfg: ScannetppBlurCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        if cfg.near > 0:
            self.near = cfg.near
        if cfg.far > 0:
            self.far = cfg.far

        if cfg.scene_list:
            self._scenes = list(cfg.scene_list)
        else:
            self._scenes = self._discover_scenes()

    def _discover_scenes(self) -> list[str]:
        scenes = []
        for root in self.cfg.roots:
            root = Path(root)
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                if not (entry / "poses_iphone.json").exists():
                    continue
                sharp_json = entry / "poses_dslr_sharp.json"
                if not sharp_json.exists():
                    continue
                try:
                    sharp_data = json.load(open(sharp_json))
                    if len(sharp_data) < self.cfg.min_sharp_frames:
                        continue
                except Exception:
                    continue
                if (entry / "iphone_frames").exists() and (entry / "dslr_sharp_frames").exists():
                    scenes.append(entry.name)
        return scenes

    def _load_image(self, path: Path) -> Float[Tensor, "3 h w"]:
        img = Image.open(path).convert("RGB")
        h, w = self.cfg.input_image_shape
        img = img.resize((w, h), Image.LANCZOS)
        return self.to_tensor(img)

    def get_bound(self, bound: Literal["near", "far"],
                  num_views: int) -> Float[Tensor, "view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        scenes = list(self._scenes)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank  = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
            scenes = [s for i, s in enumerate(scenes) if i % world == rank]

        if worker_info is not None:
            scenes = [s for i, s in enumerate(scenes)
                      if i % worker_info.num_workers == worker_info.id]

        if self.stage == "train":
            perm = torch.randperm(len(scenes)).tolist()
            scenes = [scenes[i] for i in perm]

        for scene_name in scenes:
            try:
                yield from self._iter_scene(scene_name)
            except Exception as e:
                print(f"[DatasetScannetppBlur] Skipping {scene_name}: {e}")
                continue

    def _iter_scene(self, scene_name: str):
        scene_root = None
        for root in self.cfg.roots:
            candidate = Path(root) / scene_name
            if candidate.exists():
                scene_root = candidate
                break
        if scene_root is None:
            raise FileNotFoundError(f"Scene not found: {scene_name}")

        iphone_dir     = scene_root / "iphone_frames"
        dslr_sharp_dir = scene_root / "dslr_sharp_frames"

        # ── Load matched pairs (poses_dslr_sharp.json) ────────────────────────
        # Each entry is a 1-to-1 pair: iPhone frame (blurry) ↔ DSLR frame (sharp)
        with open(scene_root / "poses_dslr_sharp.json") as f:
            dslr_data = json.load(f)

        # Build iPhone pose lookup by fid
        with open(scene_root / "poses_iphone.json") as f:
            iphone_list = json.load(f)
        iphone_by_fid = {d["fid"]: d for d in iphone_list}

        # Keep only pairs where both images exist
        valid_pairs = [
            d for d in dslr_data
            if (iphone_dir    / f"frame_{d['fid']:06d}.jpg").exists()
            and (dslr_sharp_dir / f"frame_{d['fid']:06d}.jpg").exists()
            and d["fid"] in iphone_by_fid
        ]
        if len(valid_pairs) < self.cfg.num_context_views:
            raise ValueError(
                f"Too few valid pairs in {scene_name}: {len(valid_pairs)}")

        n_views = self.cfg.num_context_views

        for _ in range(max(1, self.cfg.samples_per_scene)):
            # ── Sample n_views matched pairs ──────────────────────────────────
            sampled = random.sample(valid_pairs, n_views)

            # ── Load images ───────────────────────────────────────────────────
            try:
                ctx_images = torch.stack([
                    self._load_image(iphone_dir / f"frame_{p['fid']:06d}.jpg")
                    for p in sampled
                ])
                tgt_images = torch.stack([
                    self._load_image(dslr_sharp_dir / f"frame_{p['fid']:06d}.jpg")
                    for p in sampled
                ])
            except Exception:
                continue

            # ── Build pose tensors ────────────────────────────────────────────
            # Context uses iPhone c2w; target uses DSLR c2w (≤0.2m, ≤15° offset)
            ctx_entries = [iphone_by_fid[p["fid"]] for p in sampled]
            ctx_c2w = torch.tensor(
                np.array([e["c2w"] for e in ctx_entries], dtype=np.float32))
            tgt_c2w = torch.tensor(
                np.array([p["c2w"] for p in sampled], dtype=np.float32))

            all_c2w = torch.cat([ctx_c2w, tgt_c2w], dim=0)  # [2*N, 4, 4]

            if self.cfg.relative_pose:
                all_c2w = camera_normalization(all_c2w[0:1], all_c2w)

            ctx_extrinsics = all_c2w[:n_views]
            tgt_extrinsics = all_c2w[n_views:]

            # ── Build intrinsics (normalized by image dims) ───────────────────
            def make_K(d):
                return torch.tensor(
                    [[d["fx"], 0.0, d["cx"]],
                     [0.0, d["fy"], d["cy"]],
                     [0.0, 0.0, 1.0]], dtype=torch.float32)

            ctx_K = torch.stack([make_K(e) for e in ctx_entries])
            tgt_K = torch.stack([make_K(p) for p in sampled])

            ctx_K_norm = ctx_K.clone()
            ctx_K_norm[:, 0, :] /= ctx_entries[0]["w"]
            ctx_K_norm[:, 1, :] /= ctx_entries[0]["h"]
            tgt_K_norm = tgt_K.clone()
            tgt_K_norm[:, 0, :] /= sampled[0]["w"]
            tgt_K_norm[:, 1, :] /= sampled[0]["h"]

            # ── Overlap proxy (all pairs are matched, so overlap ≈ 1) ─────────
            overlap = torch.ones(n_views, dtype=torch.float32)

            yield {
                "context": {
                    "extrinsics": ctx_extrinsics,
                    "intrinsics": ctx_K_norm,
                    "image":      ctx_images,
                    "near":       self.get_bound("near", n_views),
                    "far":        self.get_bound("far",  n_views),
                    "index":      torch.tensor([p["fid"] for p in sampled],
                                               dtype=torch.long),
                    "overlap":    overlap,
                },
                "target": {
                    "extrinsics": tgt_extrinsics,
                    "intrinsics": tgt_K_norm,
                    "image":      tgt_images,
                    "near":       self.get_bound("near", n_views),
                    "far":        self.get_bound("far",  n_views),
                    "index":      torch.tensor([p["fid"] for p in sampled],
                                               dtype=torch.long),
                },
                "scene": scene_name,
            }

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage
