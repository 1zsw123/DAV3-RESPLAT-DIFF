"""
Dataset loader — ScanNet++ iPhone→DSLR deblur training.

Training scheme (no sharp/blurry distinction needed):
  context["image"]  = N iPhone blurry frames  (at iPhone poses)
  target["image"]   = M DSLR sharp frames     (at DSLR poses, undistorted + color-corrected)

C3G reconstructs 3DGS from blurry iPhone context, then renders at DSLR poses.
Supervision: rendered DSLR views vs real DSLR sharp frames (L1 + LPIPS).

Dataset structure (produced by scripts/prepare_scannetpp_v2.py):
  root/{scene_id}/
    iphone_frames/       frame_XXXXXX.jpg   (context, extracted from rgb.mkv)
    iphone_depth/        frame_XXXXXX.png   (context LiDAR depth, uint16 mm, 192x256, optional)
    dslr_sharp_frames/   frame_XXXXXX.jpg   (target, undistorted + color-corrected)
    poses_iphone.json    [{fid, c2w, fx, fy, cx, cy, w, h}, ...]
    poses_dslr_sharp.json [{fid, c2w, fx, fy, cx, cy, w, h, dslr_dist, dslr_angle}, ...]
"""

# iPhone LiDAR depth native resolution (from depth.bin)
_IPHONE_DEPTH_H, _IPHONE_DEPTH_W = 192, 256

import json
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
from ..misc.cam_utils import camera_normalization


@dataclass
class ScannetppDslrCfg(DatasetCfgCommon):
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
    scene_list_file: str = ""     # path to txt file with one scene ID per line
    samples_per_scene: int = 100
    num_context_views: int = 6    # iPhone context frames (max if min_context_views set)
    min_context_views: int = -1   # if > 0, randomly sample between min and num_context_views
    num_target_views: int = 2     # DSLR target frames per sample
    min_dslr_pairs: int = 20      # min DSLR-iPhone pairs to include scene
    val_samples_file: str = ""    # path to fixed val samples JSON; used in val/test stage
    max_ctx_dist: float = 0.5     # max distance (m) between iPhone and DSLR anchor
    max_ctx_angle: float = 45.0   # max angle (deg) between iPhone and DSLR anchor orientations


@dataclass
class DatasetScannetppDslrCfgWrapper:
    scannetpp_dslr: ScannetppDslrCfg


class DatasetScannetppDslr(IterableDataset):
    cfg: ScannetppDslrCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.01
    far: float  = 10.0

    def __init__(self, cfg: ScannetppDslrCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        if cfg.near > 0: self.near = cfg.near
        if cfg.far  > 0: self.far  = cfg.far

        if cfg.scene_list:
            self._scenes = list(cfg.scene_list)
        elif cfg.scene_list_file:
            with open(cfg.scene_list_file) as f:
                self._scenes = [l.strip() for l in f if l.strip()]
        else:
            self._scenes = self._discover_scenes()

    def _discover_scenes(self) -> list[str]:
        scenes = []
        for root in self.cfg.roots:
            root = Path(root)
            for entry in sorted(root.iterdir()):
                if not entry.is_dir(): continue
                if not (entry / "poses_iphone.json").exists(): continue
                if not (entry / "poses_dslr_sharp.json").exists(): continue
                if not (entry / "iphone_frames").exists(): continue
                if not (entry / "dslr_sharp_frames").exists(): continue
                try:
                    dslr = json.load(open(entry / "poses_dslr_sharp.json"))
                    if len(dslr) < self.cfg.min_dslr_pairs:
                        continue
                except Exception:
                    continue
                scenes.append(entry.name)
        return scenes

    def _load_mask(self, path: Path, h: int, w: int) -> Float[Tensor, "1 h w"]:
        out_h, out_w = self.cfg.original_image_shape
        if path.exists():
            mask = Image.open(path).convert("L")
            t = self.to_tensor(mask.resize((out_w, out_h), Image.NEAREST))
        else:
            t = torch.ones(1, out_h, out_w)
        return (t > 0.5).float()

    def _load_image(self, path: Path) -> Float[Tensor, "3 h w"]:
        h, w = self.cfg.input_image_shape
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img.resize((w, h), Image.LANCZOS))

    def _load_target_image(self, path: Path, h: int, w: int) -> Float[Tensor, "3 h w"]:
        out_h, out_w = self.cfg.original_image_shape
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img.resize((out_w, out_h), Image.LANCZOS))

    def _load_depth(self, path: Path, h: int, w: int) -> Float[Tensor, "h w"] | None:
        """Load 16-bit PNG depth (mm) → float32 metres, resized to original_image_shape. Returns None if missing."""
        if not path.exists():
            return None
        out_h, out_w = self.cfg.original_image_shape
        depth_mm  = np.array(Image.open(path), dtype=np.uint16)
        depth_pil = Image.fromarray(depth_mm, mode="I;16").resize((out_w, out_h), Image.NEAREST)
        depth_m   = np.array(depth_pil, dtype=np.float32) / 1000.0
        return torch.from_numpy(depth_m)   # [H, W]

    def get_bound(self, bound: Literal["near", "far"], n: int) -> Float[Tensor, "view"]:
        return repeat(torch.tensor(getattr(self, bound), dtype=torch.float32), "-> v", v=n)

    def __iter__(self):
        # Val/test: use fixed sample list if provided
        if self.stage != "train" and self.cfg.val_samples_file:
            yield from self._iter_fixed_samples()
            return

        worker_info = torch.utils.data.get_worker_info()
        scenes = list(self._scenes)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
            scenes = [s for i, s in enumerate(scenes) if i % world == rank]
        if worker_info is not None:
            scenes = [s for i, s in enumerate(scenes)
                      if i % worker_info.num_workers == worker_info.id]
        if self.stage == "train":
            scenes = [scenes[i] for i in torch.randperm(len(scenes)).tolist()]

        for scene_name in scenes:
            try:
                yield from self._iter_scene(scene_name)
            except Exception as e:
                print(f"[DatasetScannetppDslr] Skipping {scene_name}: {e}")

    def _iter_fixed_samples(self):
        """Iterate over a pre-generated fixed sample list (for reproducible val/test)."""
        samples = json.load(open(self.cfg.val_samples_file))
        n_ctx = self.cfg.num_context_views
        n_tgt = self.cfg.num_target_views

        for entry in samples:
            scene_name = entry["scene"]
            ctx_fids   = entry["ctx_fids"][:n_ctx]   # take first n_ctx from stored 24
            tgt_fids   = entry["tgt_fids"][:n_tgt]   # take first n_tgt from stored 4

            scene_root = None
            for root in self.cfg.roots:
                c = Path(root) / scene_name
                if c.exists(): scene_root = c; break
            if scene_root is None:
                continue

            iphone_dir = scene_root / "iphone_frames"
            dslr_dir   = scene_root / "dslr_sharp_frames"
            mask_dir   = scene_root / "dslr_masks"

            try:
                with open(scene_root / "poses_iphone.json") as f:
                    iphone_poses = {p["fid"]: p for p in json.load(f)}
                with open(scene_root / "poses_dslr_sharp.json") as f:
                    dslr_poses = {p["fid"]: p for p in json.load(f)}

                ctx_sample = [iphone_poses[fid] for fid in ctx_fids if fid in iphone_poses]
                tgt_sample = [dslr_poses[fid]   for fid in tgt_fids  if fid in dslr_poses]

                if len(ctx_sample) < n_ctx or len(tgt_sample) < n_tgt:
                    continue

                def make_K(d):
                    return torch.tensor([[d["fx"], 0, d["cx"]],
                                         [0, d["fy"], d["cy"]],
                                         [0, 0, 1]], dtype=torch.float32)

                ctx_imgs = torch.stack([
                    self._load_image(iphone_dir / f"frame_{p['fid']:06d}.jpg")
                    for p in ctx_sample
                ])
                ctx_c2w = torch.tensor(np.array([p["c2w"] for p in ctx_sample], np.float32))
                ctx_K   = torch.stack([make_K(p) for p in ctx_sample])
                ctx_Kn  = ctx_K.clone()
                for i, p in enumerate(ctx_sample):
                    ctx_Kn[i, 0, :] /= p["w"]
                    ctx_Kn[i, 1, :] /= p["h"]
                ctx_fids_t = torch.tensor([p["fid"] for p in ctx_sample], dtype=torch.long)

                tgt_imgs = torch.stack([
                    self._load_target_image(dslr_dir / f"frame_{p['fid']:06d}.jpg",
                                            p["h"], p["w"])
                    for p in tgt_sample
                ])
                tgt_masks = torch.stack([
                    self._load_mask(mask_dir / f"frame_{p['fid']:06d}.png",
                                    p["h"], p["w"])
                    for p in tgt_sample
                ])
                tgt_c2w = torch.tensor(np.array([p["c2w"] for p in tgt_sample], np.float32))
                tgt_K   = torch.stack([make_K(p) for p in tgt_sample])
                tgt_Kn  = tgt_K.clone()
                for i, p in enumerate(tgt_sample):
                    tgt_Kn[i, 0, :] /= p["w"]
                    tgt_Kn[i, 1, :] /= p["h"]
                tgt_fids_t = torch.tensor([p["fid"] for p in tgt_sample], dtype=torch.long)

                depth_dir = scene_root / "dslr_depth"
                tgt_depths_list = [
                    self._load_depth(depth_dir / f"frame_{p['fid']:06d}.png", p["h"], p["w"])
                    for p in tgt_sample
                ]
                tgt_depths = (torch.stack(tgt_depths_list)
                              if all(d is not None for d in tgt_depths_list)
                              else None)

            except Exception as e:
                print(f"[DatasetScannetppDslr] Skipping fixed sample {scene_name}: {e}")
                continue

            if self.cfg.relative_pose:
                all_c2w = torch.cat([ctx_c2w, tgt_c2w], dim=0)
                all_c2w = camera_normalization(all_c2w[0:1], all_c2w)
                ctx_c2w, tgt_c2w = all_c2w[:n_ctx], all_c2w[n_ctx:]

            target_dict = {
                "extrinsics": tgt_c2w,
                "intrinsics": tgt_Kn,
                "image":      tgt_imgs,
                "mask":       tgt_masks,
                "near":       self.get_bound("near", n_tgt),
                "far":        self.get_bound("far",  n_tgt),
                "index":      tgt_fids_t,
            }
            if tgt_depths is not None:
                target_dict["depth"] = tgt_depths

            yield {
                "context": {
                    "extrinsics": ctx_c2w,
                    "intrinsics": ctx_Kn,
                    "image":      ctx_imgs,
                    "near":       self.get_bound("near", n_ctx),
                    "far":        self.get_bound("far",  n_ctx),
                    "index":      ctx_fids_t,
                    "overlap":    torch.ones(n_ctx, dtype=torch.float32),
                },
                "target": target_dict,
                "scene": scene_name,
            }

    def _iter_scene(self, scene_name: str):
        scene_root = None
        for root in self.cfg.roots:
            c = Path(root) / scene_name
            if c.exists(): scene_root = c; break
        if scene_root is None:
            raise FileNotFoundError(scene_name)

        iphone_dir = scene_root / "iphone_frames"
        dslr_dir   = scene_root / "dslr_sharp_frames"
        mask_dir   = scene_root / "dslr_masks"

        with open(scene_root / "poses_iphone.json") as f:
            iphone_poses = json.load(f)
        with open(scene_root / "poses_dslr_sharp.json") as f:
            dslr_poses = json.load(f)

        # filter to frames that actually exist on disk
        iphone_valid = [p for p in iphone_poses
                        if (iphone_dir / f"frame_{p['fid']:06d}.jpg").exists()]
        dslr_valid   = [p for p in dslr_poses
                        if (dslr_dir / f"frame_{p['fid']:06d}.jpg").exists()]

        if self.cfg.min_context_views > 0 and self.stage == "train":
            n_ctx = random.randint(self.cfg.min_context_views, self.cfg.num_context_views)
        else:
            n_ctx = self.cfg.num_context_views
        n_tgt = self.cfg.num_target_views

        if len(iphone_valid) < n_ctx:
            raise ValueError(f"Too few iPhone frames: {len(iphone_valid)}")
        if len(dslr_valid) < n_tgt:
            raise ValueError(f"Too few DSLR frames: {len(dslr_valid)}")

        # precompute positions and orientations for context filtering
        iphone_c2w  = np.array([p["c2w"] for p in iphone_valid], np.float32)
        iphone_pos  = iphone_c2w[:, :3, 3]
        iphone_fwd  = iphone_c2w[:, :3, 2]   # forward axis (col 2 of R)
        dslr_pos    = np.array([np.array(p["c2w"])[:3, 3] for p in dslr_valid], np.float32)

        def make_K(d):
            return torch.tensor([[d["fx"], 0, d["cx"]],
                                 [0, d["fy"], d["cy"]],
                                 [0, 0, 1]], dtype=torch.float32)

        max_ctx_dist  = self.cfg.max_ctx_dist
        max_ctx_angle = self.cfg.max_ctx_angle

        for _ in range(max(1, self.cfg.samples_per_scene)):
            # sample 1 DSLR target, find iPhone frames nearby in position and orientation
            dslr_anchor  = random.choice(dslr_valid)
            anchor_c2w   = np.array(dslr_anchor["c2w"], np.float32)
            anchor_pos   = anchor_c2w[:3, 3]
            anchor_fwd   = anchor_c2w[:3, 2]

            # distance and angle filters
            dists  = np.linalg.norm(iphone_pos - anchor_pos, axis=1)
            cos_a  = np.clip((iphone_fwd * anchor_fwd).sum(axis=1), -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_a))
            mask   = (dists <= max_ctx_dist) & (angles <= max_ctx_angle)

            if mask.sum() >= n_ctx:
                cand_idx   = np.where(mask)[0]
                cand_dists = dists[cand_idx]
                sorted_idx = cand_idx[np.argsort(cand_dists)]
                nearest_idx = sorted_idx[:n_ctx]
            else:
                # fallback: take nearest regardless of threshold
                nearest_idx = np.argsort(dists)[:n_ctx]
            ctx_sample  = [iphone_valid[i] for i in nearest_idx]

            # Select n_tgt DSLR targets: nearest by position to anchor (spatially coherent)
            dslr_dists  = np.linalg.norm(dslr_pos - anchor_pos, axis=1)
            dslr_near_idx = np.argsort(dslr_dists)[:n_tgt]
            dslr_sample = [dslr_valid[i] for i in dslr_near_idx]

            try:
                # ── context: iPhone blurry frames ──────────────────────────
                ctx_imgs = torch.stack([
                    self._load_image(iphone_dir / f"frame_{p['fid']:06d}.jpg")
                    for p in ctx_sample
                ])
                ctx_c2w = torch.tensor(
                    np.array([p["c2w"] for p in ctx_sample], np.float32))
                ctx_K   = torch.stack([make_K(p) for p in ctx_sample])
                ctx_Kn  = ctx_K.clone()
                for i, p in enumerate(ctx_sample):
                    ctx_Kn[i, 0, :] /= p["w"]
                    ctx_Kn[i, 1, :] /= p["h"]
                ctx_fids = torch.tensor([p["fid"] for p in ctx_sample], dtype=torch.long)

                # ── target: DSLR sharp frames + valid masks ─────────────────
                # h, w from pose JSON (720×480 native DSLR resolution, 3:2 ratio)
                tgt_imgs = torch.stack([
                    self._load_target_image(dslr_dir / f"frame_{p['fid']:06d}.jpg",
                                            p["h"], p["w"])
                    for p in dslr_sample
                ])
                tgt_masks = torch.stack([
                    self._load_mask(mask_dir / f"frame_{p['fid']:06d}.png",
                                    p["h"], p["w"])
                    for p in dslr_sample
                ])
                tgt_c2w = torch.tensor(
                    np.array([p["c2w"] for p in dslr_sample], np.float32))
                tgt_K   = torch.stack([make_K(p) for p in dslr_sample])
                tgt_Kn  = tgt_K.clone()
                for i, p in enumerate(dslr_sample):
                    tgt_Kn[i, 0, :] /= p["w"]
                    tgt_Kn[i, 1, :] /= p["h"]
                tgt_fids = torch.tensor([p["fid"] for p in dslr_sample], dtype=torch.long)

                # ── target: depth maps (optional, from mesh raycasting) ────
                dslr_depth_dir = scene_root / "dslr_depth"
                tgt_depths_list = [
                    self._load_depth(dslr_depth_dir / f"frame_{p['fid']:06d}.png", p["h"], p["w"])
                    for p in dslr_sample
                ]
                tgt_depths = (torch.stack(tgt_depths_list)   # [n_tgt, H, W]
                              if all(d is not None for d in tgt_depths_list)
                              else None)

                # ── context: iPhone LiDAR depth (optional) ─────────────────
                iphone_depth_dir = scene_root / "iphone_depth"
                ctx_depths_list = [
                    self._load_depth(iphone_depth_dir / f"frame_{p['fid']:06d}.png",
                                     _IPHONE_DEPTH_H, _IPHONE_DEPTH_W)
                    for p in ctx_sample
                ]
                ctx_depths = (torch.stack(ctx_depths_list)   # [n_ctx, H, W]
                              if all(d is not None for d in ctx_depths_list)
                              else None)

            except Exception:
                continue

            # ── relative pose normalisation (w.r.t. first context view) ────
            if self.cfg.relative_pose:
                all_c2w = torch.cat([ctx_c2w, tgt_c2w], dim=0)
                all_c2w = camera_normalization(all_c2w[0:1], all_c2w)
                ctx_c2w, tgt_c2w = all_c2w[:n_ctx], all_c2w[n_ctx:]

            target_dict = {
                "extrinsics": tgt_c2w,
                "intrinsics": tgt_Kn,
                "image":      tgt_imgs,
                "mask":       tgt_masks,
                "near":       self.get_bound("near", n_tgt),
                "far":        self.get_bound("far",  n_tgt),
                "index":      tgt_fids,
            }
            if tgt_depths is not None:
                target_dict["depth"] = tgt_depths

            context_dict = {
                "extrinsics": ctx_c2w,
                "intrinsics": ctx_Kn,
                "image":      ctx_imgs,
                "near":       self.get_bound("near", n_ctx),
                "far":        self.get_bound("far",  n_ctx),
                "index":      ctx_fids,
                "overlap":    torch.ones(n_ctx, dtype=torch.float32),
            }
            if ctx_depths is not None:
                context_dict["depth"] = ctx_depths

            yield {
                "context": context_dict,
                "target": target_dict,
                "scene": scene_name,
            }

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None: return "test"
        if self.stage == "val": return "test"
        return self.stage
