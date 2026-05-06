"""
Dataset loader for I2-SLAM ScanNet (4 scenes), single-camera multi-view bundles.

Per-scene structure:
  root/{scene_id}/
    gt/color_NNNNN.png        (640x480 blurry input — used as ctx)
    gt/depth_NNNNN.npy        (1, 480, 640) float32 metres, 0=hole/invalid
    vdiff/color_NNNNN.png     (VDiff-deblurred pseudo-GT — used as tgt)
    poses.npy                 (M, 4, 4) float32 c2w (i2slam frame-aligned)
    intrinsic_color.txt       4x4 (1296x968 native; we scale to input shape)

Training scheme (deblur):
  context: N consecutive blurry frames at GT poses
  target : same N frames, target image = VDiff pseudo-GT (1-to-1 deblur supervision)
  target.depth: ScanNet GT depth at the same N frames (0=invalid → masked in loss)
"""

from __future__ import annotations
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
class ScannetI2SlamCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]

    near: float = 0.1
    far: float  = 100.0

    baseline_min: float = 0.0
    baseline_max: float = 1.0e10
    max_fov: float = 180.0
    make_baseline_1: bool = False
    augment: bool = False
    relative_pose: bool = False
    skip_bad_shape: bool = True

    # Native i2slam color resolution (W, H stored as [H, W] convention).
    # scaled K is computed from this.
    native_color_w: int = 1296   # ScanNet color sensor native width
    native_color_h: int = 968    # ScanNet color sensor native height

    # Scene split — held-out scenes go to val. Default: 0785_00 → val, others train.
    train_scenes: list[str] = field(default_factory=lambda: [
        "scene0024_01", "scene0031_00", "scene0736_00",
    ])
    val_scenes: list[str]   = field(default_factory=lambda: [
        "scene0785_00",
    ])

    # Sampling
    samples_per_scene: int = 100      # train samples per epoch per scene
    val_samples_per_scene: int = 8
    num_context_views: int = 6
    frame_stride: int = 1             # 1 = consecutive; >1 = wider baseline


@dataclass
class DatasetScannetI2SlamCfgWrapper:
    scannet_i2slam: ScannetI2SlamCfg


class DatasetScannetI2Slam(IterableDataset):
    cfg: ScannetI2SlamCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(self, cfg: ScannetI2SlamCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        scenes = cfg.train_scenes if stage == "train" else cfg.val_scenes

        # Pre-load metadata per scene: poses, K, n_frames
        self._scenes: dict[str, dict] = {}
        for s in scenes:
            for root in cfg.roots:
                root = Path(root)
                sd = root / s
                if not (sd / "poses.npy").exists():
                    continue
                poses = np.load(sd / "poses.npy")          # (M, 4, 4) c2w
                K = np.loadtxt(sd / "intrinsic_color.txt")[:3, :3].astype(np.float32)
                # Scene-level translation scale: median ‖ t_i - t_0 ‖ over the
                # whole trajectory, used to put the scene roughly at unit metric
                # scale (≈ Replica training distribution). Computing it per-
                # ctx-pair (Replica's make_baseline_1) is unsafe here because
                # ScanNet samples 6 *consecutive* frames whose baseline is
                # 0.01-0.05m → amplifies translations 20-100x → catastrophic OOD.
                t = poses[:, :3, 3]
                d = np.linalg.norm(t - t[0:1], axis=-1)
                scene_scale = float(np.median(d[1:])) if len(d) > 1 else 1.0
                scene_scale = max(scene_scale, 1e-3)
                self._scenes[s] = {
                    "root": sd,
                    "poses": poses,
                    "K_native": K,    # at (cfg.native_color_w, cfg.native_color_h)
                    "n": int(poses.shape[0]),
                    "scene_scale": scene_scale,
                }
                break

    # ─────────────────────────────────────────────────────────────────────
    def _scaled_K(self, K_native: np.ndarray) -> torch.Tensor:
        """Scale 1296x968 intrinsics to input_image_shape, return normalized 3x3."""
        H_t, W_t = self.cfg.input_image_shape
        sx = W_t / self.cfg.native_color_w
        sy = H_t / self.cfg.native_color_h
        K = K_native.copy()
        K[0] *= sx; K[1] *= sy
        # normalise
        K_norm = K.copy()
        K_norm[0] /= W_t; K_norm[1] /= H_t
        return torch.from_numpy(K_norm).float()

    def _load_color(self, path: Path) -> Tensor:
        H_t, W_t = self.cfg.input_image_shape
        img = Image.open(path).convert("RGB").resize((W_t, H_t), Image.LANCZOS)
        return self.to_tensor(img)

    def _load_depth(self, path: Path) -> Tensor:
        """Load (1, 480, 640) float32 metres, resize to input shape with NEAREST
        (preserve hole=0 markers). Returns [H_t, W_t] tensor."""
        H_t, W_t = self.cfg.input_image_shape
        d = np.load(path)
        if d.ndim == 3:    # (1, H, W) → (H, W)
            d = d[0]
        d_t = torch.from_numpy(d).float()[None, None]   # [1, 1, H, W]
        d_t = F.interpolate(d_t, size=(H_t, W_t), mode="nearest")
        return d_t.squeeze(0).squeeze(0)                # [H_t, W_t]

    def get_bound(self, bound: Literal["near", "far"], n: int) -> Float[Tensor, "v"]:
        return repeat(torch.tensor(getattr(self.cfg, bound), dtype=torch.float32), "-> v", v=n)

    # ─────────────────────────────────────────────────────────────────────
    def _sample_one(self, scene_id: str, scene: dict, rng: random.Random):
        n = scene["n"]
        N_ctx = self.cfg.num_context_views
        stride = self.cfg.frame_stride
        max_start = n - (N_ctx - 1) * stride - 1
        if max_start < 0:
            return None
        start = rng.randint(0, max_start)
        idxs = [start + i * stride for i in range(N_ctx)]

        root = scene["root"]
        try:
            blur_imgs  = torch.stack([self._load_color(root / "gt"    / f"color_{i:05d}.png") for i in idxs])
            sharp_imgs = torch.stack([self._load_color(root / "vdiff" / f"color_{i:05d}.png") for i in idxs])
            depths     = torch.stack([self._load_depth(root / "gt"    / f"depth_{i:05d}.npy") for i in idxs])
        except (FileNotFoundError, OSError):
            return None

        # ScanNet .sens stores poses as C2W (verified empirically: trajectory
        # smoothness test shows mean inter-frame step 13.8mm under c2w
        # interpretation vs 48.6mm under w2c, consistent with handheld 30fps).
        # The SensorData.py variable name `camera_to_world` matches the format.
        c2w = torch.from_numpy(scene["poses"][idxs].astype(np.float32))         # [V, 4, 4] C2W

        # NOTE: Replica training uses make_baseline_1=False with absolute-metre
        # GT poses; encoder._normalise_poses median-clamp(0.1) handles small
        # ctx baselines uniformly. Match that here — no dataset-side scaling.

        K_norm = self._scaled_K(scene["K_native"])
        K = K_norm.unsqueeze(0).expand(N_ctx, 3, 3).contiguous()

        idx_t = torch.tensor(idxs, dtype=torch.long)
        ctx = {
            "extrinsics":  c2w,
            "intrinsics":  K,
            "image":       blur_imgs,
            "sharp_image": sharp_imgs,
            "near":        self.get_bound("near", N_ctx),
            "far":         self.get_bound("far",  N_ctx),
            "index":       idx_t,
            "overlap":     torch.ones(N_ctx, dtype=torch.float32),
        }
        tgt = {
            "extrinsics":  c2w.clone(),
            "intrinsics":  K.clone(),
            "image":       sharp_imgs,
            "depth":       depths,                         # [V, H, W] metres, 0=invalid
            "near":        self.get_bound("near", N_ctx),
            "far":         self.get_bound("far",  N_ctx),
            "index":       idx_t.clone(),
        }
        return {"context": ctx, "target": tgt, "scene": scene_id}

    def __iter__(self):
        scenes = list(self._scenes.items())
        if self.stage == "train":
            random.shuffle(scenes)
            n_per = self.cfg.samples_per_scene
            rng = random.Random()
        else:
            n_per = self.cfg.val_samples_per_scene
            # deterministic val
            rng = random.Random(42)

        for scene_id, scene in scenes:
            for _ in range(n_per):
                sample = self._sample_one(scene_id, scene, rng)
                if sample is not None:
                    yield sample

    @property
    def data_stage(self) -> Stage:
        if self.stage == "val":
            return "test"
        return self.stage
