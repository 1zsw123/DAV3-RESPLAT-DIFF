"""
Unified dataset loader for COLMAP/LLFF-format benchmark datasets:
  - Deblur-NeRF  (real_camera_motion_blur, real_defocus_blur)
  - ExBlur-NeRF  (exblur_release)
  - NeRF-LLFF    (nerf_llff_data)

All three share the poses_bounds.npy format (N, 17):
  poses = data[:, :15].reshape(N, 3, 5)   C2W in OpenCV convention + H/W/focal
  bds   = data[:, 15:]                     per-frame near/far

Coordinate convention: poses_bounds stores C2W in OpenCV (x=right, y=down, z=forward).
We convert to OpenGL/NeRF (x=right, y=up, z=backward) by negating columns 1 and 2.
"""

import os
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
class ColmapCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]                     # list of root dirs to scan for scenes

    # --- image directories ---
    image_dir: str = "images_4"           # blurry context images (downsampled dir)
    sharp_dir: str = ""                   # if set, use for target GT (ExBlur-NeRF)

    # --- split logic ---
    test_hold: int = 0                    # if >0, every N-th frame is test (Deblur-NeRF / LLFF)
    test_txt: bool = False                # if True, read train.txt / test.txt

    # --- scene bounds ---
    near: float = 0.01
    far: float = 10.0

    # --- view counts ---
    num_context_views: int = 10           # train frames used as context
    num_target_views: int = 1            # test frames per sample

    # --- standard flags ---
    baseline_min: float = 1e-3
    baseline_max: float = 1e10
    max_fov: float = 120.0
    make_baseline_1: bool = False
    augment: bool = False
    relative_pose: bool = True
    skip_bad_shape: bool = False

    # --- scene selection ---
    scene_list: list[str] = field(default_factory=list)

    # --- how many samples to draw per scene per epoch ---
    samples_per_scene: int = 50


@dataclass
class DatasetColmapCfgWrapper:
    colmap: ColmapCfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_poses_bounds(path: str):
    """
    Load poses_bounds.npy and return (c2w_opengl, H, W, focal, bds).

    Returns
    -------
    c2w   : np.ndarray  (N, 4, 4)  float32, C2W in OpenGL convention
    H     : int         full-resolution image height
    W     : int         full-resolution image width
    focal : float       focal length for full-resolution images
    bds   : np.ndarray  (N, 2)     per-frame near/far from COLMAP
    """
    data = np.load(path)                   # (N, 17)
    poses = data[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
    bds   = data[:, 15:]                   # (N, 2)

    H     = int(poses[0, 0, 4])
    W     = int(poses[0, 1, 4])
    focal = float(poses[0, 2, 4])

    # Extract 3x4 C2W block (OpenCV convention)
    c2w_opencv = poses[:, :3, :4]           # (N, 3, 4)

    # Build 4x4 homogeneous matrices
    N = c2w_opencv.shape[0]
    c2w = np.zeros((N, 4, 4), dtype=np.float32)
    c2w[:, :3, :4] = c2w_opencv
    c2w[:,  3,  3] = 1.0

    # OpenCV → OpenGL: flip y and z axes
    c2w[:, :3, 1:3] *= -1

    return c2w, H, W, focal, bds


def _infer_downsample_factor(image_dir: str) -> float:
    """
    Infer the downsampling factor from the image directory name.
    E.g. "images_4" → 4.0, "images_4_vdiff" → 4.0, "images_8" → 8.0, "images" → 1.0.
    Finds the first integer following an underscore.
    """
    import re
    name = Path(image_dir).name
    m = re.search(r'_(\d+)', name)
    if m:
        return float(m.group(1))
    return 1.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetColmap(IterableDataset):
    cfg: ColmapCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.01
    far: float = 10.0

    def __init__(
        self,
        cfg: ColmapCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        if cfg.near > 0:
            self.near = cfg.near
        if cfg.far > 0:
            self.far = cfg.far

        self._downsample = _infer_downsample_factor(cfg.image_dir)

        # Discover scenes
        if cfg.scene_list:
            self._scenes = list(cfg.scene_list)
        else:
            self._scenes = self._discover_scenes()

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------

    def _discover_scenes(self) -> list[str]:
        scenes = []
        for root in self.cfg.roots:
            root = Path(root)
            if not root.exists():
                continue
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                if (entry / "poses_bounds.npy").exists():
                    scenes.append(entry.name)
        return scenes

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> Float[Tensor, "3 h w"]:
        img = Image.open(path).convert("RGB")
        target_h, target_w = self.cfg.input_image_shape
        img = img.resize((target_w, target_h), Image.LANCZOS)
        return self.to_tensor(img)

    # ------------------------------------------------------------------
    # Bound helpers
    # ------------------------------------------------------------------

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Float[Tensor, "view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        scenes = list(self._scenes)

        # Shard scenes across DDP ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            scenes = [s for i, s in enumerate(scenes) if i % world_size == rank]

        # Shard scenes across dataloader workers within each rank
        if worker_info is not None:
            scenes = [
                s for i, s in enumerate(scenes)
                if i % worker_info.num_workers == worker_info.id
            ]

        # Shuffle during training
        if self.stage == "train":
            scenes = [scenes[i] for i in torch.randperm(len(scenes)).tolist()]

        for scene_name in scenes:
            try:
                yield from self._iter_scene(scene_name)
            except Exception as e:
                print(f"[DatasetColmap] Skipping scene {scene_name}: {e}")
                continue

    def _iter_scene(self, scene_name: str):
        # Locate scene root
        scene_root = None
        for root in self.cfg.roots:
            candidate = Path(root) / scene_name
            if candidate.exists():
                scene_root = candidate
                break
        if scene_root is None:
            raise FileNotFoundError(f"Scene not found: {scene_name}")

        # Load poses_bounds
        poses_bounds_path = scene_root / "poses_bounds.npy"
        if not poses_bounds_path.exists():
            raise FileNotFoundError(f"poses_bounds.npy missing: {scene_root}")

        c2w_all, H_full, W_full, focal_full, bds = _load_poses_bounds(str(poses_bounds_path))
        N = c2w_all.shape[0]

        # Locate image directory
        img_dir = scene_root / self.cfg.image_dir
        if not img_dir.exists():
            raise FileNotFoundError(f"Image dir missing: {img_dir}")

        # Collect image files sorted by name
        img_files = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ], key=lambda p: p.name)

        if len(img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")

        # Match image count to pose count (trim if necessary)
        n_imgs = min(len(img_files), N)
        img_files = img_files[:n_imgs]
        c2w_all = c2w_all[:n_imgs]
        bds = bds[:n_imgs]

        # Read actual image resolution from first image
        sample_img = Image.open(str(img_files[0]))
        img_W_actual, img_H_actual = sample_img.size  # PIL: (width, height)
        sample_img.close()

        # Compute effective focal length for the loaded images
        # poses_bounds stores focal for full-res; images are downsampled by factor
        focal_eff = focal_full / self._downsample

        # Normalized intrinsics matrix (same for all frames)
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = focal_eff / img_W_actual   # fx / W
        K[1, 1] = focal_eff / img_H_actual   # fy / H
        K[0, 2] = 0.5                         # cx / W  (principal point at center)
        K[1, 2] = 0.5                         # cy / H

        # Build intrinsics tensor (same K for all frames)
        intrinsics_all = torch.from_numpy(
            np.stack([K] * n_imgs, axis=0)
        )  # (N, 3, 3)

        # Build extrinsics tensor (C2W in OpenGL convention, 4x4)
        extrinsics_all = torch.from_numpy(c2w_all)  # (N, 4, 4)

        # Compute train / test split indices
        indices = list(range(n_imgs))

        if self.cfg.test_txt:
            # Read train.txt and test.txt
            train_txt = scene_root / "train.txt"
            test_txt  = scene_root / "test.txt"
            if not train_txt.exists() or not test_txt.exists():
                raise FileNotFoundError(f"train.txt / test.txt missing in {scene_root}")
            # Lines may be integer indices ("0", "7") or filenames ("000.png")
            def _parse_split_line(line: str, img_names: list[str]) -> int:
                s = line.strip()
                if not s:
                    return -1
                try:
                    return int(s)
                except ValueError:
                    stem = Path(s).stem  # "000.png" → "000"
                    for i, name in enumerate(img_names):
                        if Path(name).stem == stem:
                            return i
                    return -1

            with open(train_txt) as f:
                train_indices = [_parse_split_line(l, img_files) for l in f]
                train_indices = [i for i in train_indices if i >= 0]
            with open(test_txt) as f:
                test_indices = [_parse_split_line(l, img_files) for l in f]
                test_indices = [i for i in test_indices if i >= 0]
            # Clamp to valid range
            train_indices = [i for i in train_indices if i < n_imgs]
            test_indices  = [i for i in test_indices  if i < n_imgs]
        elif self.cfg.test_hold > 0:
            h = self.cfg.test_hold
            test_indices  = [i for i in indices if i % h == 0]
            train_indices = [i for i in indices if i % h != 0]
        else:
            # No split: all frames serve as both train and test
            train_indices = indices
            test_indices  = indices

        if len(train_indices) == 0:
            raise ValueError(f"No training frames for scene {scene_name}")
        if len(test_indices) == 0:
            raise ValueError(f"No test frames for scene {scene_name}")

        # Locate sharp GT directory (optional, ExBlur-NeRF)
        sharp_gt_dir = None
        if self.cfg.sharp_dir:
            candidate = scene_root / self.cfg.sharp_dir
            if candidate.exists():
                sharp_gt_dir = candidate

        # Clamp num_context_views to available training frames
        n_ctx = min(self.cfg.num_context_views, len(train_indices))
        n_tgt = self.cfg.num_target_views

        n_samples = max(1, self.cfg.samples_per_scene)
        for _ in range(n_samples):
            # Sample context from train split, target from test split
            if len(train_indices) >= n_ctx:
                ctx_indices = random.sample(train_indices, n_ctx)
            else:
                ctx_indices = list(train_indices)
            ctx_indices.sort()

            if len(test_indices) >= n_tgt:
                tgt_indices = random.sample(test_indices, n_tgt)
            else:
                tgt_indices = list(test_indices)
            tgt_indices.sort()

            try:
                # ── Load context images (blurry) ──────────────────────────────
                ctx_images = torch.stack([
                    self._load_image(str(img_files[i])) for i in ctx_indices
                ])  # (n_ctx, 3, H, W)

                # ── Load target images (sharp GT or blurry fallback) ──────────
                tgt_image_list = []
                for i in tgt_indices:
                    if sharp_gt_dir is not None:
                        # Same filename as blurry image, but in sharp_dir
                        sharp_path = sharp_gt_dir / img_files[i].name
                        if sharp_path.exists():
                            tgt_image_list.append(self._load_image(str(sharp_path)))
                            continue
                    # Fallback: use blurry image as target
                    tgt_image_list.append(self._load_image(str(img_files[i])))
                tgt_images = torch.stack(tgt_image_list)  # (n_tgt, 3, H, W)

            except Exception as e:
                continue

            # ── Build pose/intrinsics tensors ─────────────────────────────────
            ctx_idx_t = torch.tensor(ctx_indices, dtype=torch.long)
            tgt_idx_t = torch.tensor(tgt_indices, dtype=torch.long)

            ctx_extrinsics = extrinsics_all[ctx_idx_t]   # (n_ctx, 4, 4)
            tgt_extrinsics = extrinsics_all[tgt_idx_t]   # (n_tgt, 4, 4)
            ctx_intrinsics = intrinsics_all[ctx_idx_t]   # (n_ctx, 3, 3)
            tgt_intrinsics = intrinsics_all[tgt_idx_t]   # (n_tgt, 3, 3)

            all_sel_ext = torch.cat([ctx_extrinsics, tgt_extrinsics], dim=0)

            # ── Baseline scaling ──────────────────────────────────────────────
            scale = 1.0
            if self.cfg.make_baseline_1:
                a = ctx_extrinsics[0,  :3, 3]
                b = ctx_extrinsics[-1, :3, 3]
                scale = float((a - b).norm())
                if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                    continue
                all_sel_ext = all_sel_ext.clone()
                all_sel_ext[:, :3, 3] /= scale

            # ── Relative pose normalisation ───────────────────────────────────
            if self.cfg.relative_pose:
                all_sel_ext = camera_normalization(all_sel_ext[0:1], all_sel_ext)

            ctx_extrinsics = all_sel_ext[:n_ctx]
            tgt_extrinsics = all_sel_ext[n_ctx:]

            yield {
                "context": {
                    "extrinsics": ctx_extrinsics,
                    "intrinsics": ctx_intrinsics,
                    "image":      ctx_images,
                    "near":       self.get_bound("near", n_ctx) / scale,
                    "far":        self.get_bound("far",  n_ctx) / scale,
                    "index":      ctx_idx_t,
                },
                "target": {
                    "extrinsics": tgt_extrinsics,
                    "intrinsics": tgt_intrinsics,
                    "image":      tgt_images,
                    "near":       self.get_bound("near", n_tgt) / scale,
                    "far":        self.get_bound("far",  n_tgt) / scale,
                    "index":      tgt_idx_t,
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
