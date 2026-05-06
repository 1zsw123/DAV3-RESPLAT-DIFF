"""
Dataset loader for DeblurNeRF (real_camera_motion_blur) with EVSSM→CoMoGaussian
pseudo-GT and pseudo-depth supervision.

Map-style (Dataset, not IterableDataset) so DDP DistributedSampler shards
properly across ranks during val. Without this, val_loader yields 0 batches
under DDP and validation_step is never called.

Per-scene structure:
  /scratch-shared/qzhang1/datasets/deblurnerf/real_camera_motion_blur/{scene}/
    images_4/000.png           (600x400 blurry input — used as ctx)
    poses_bounds.npy           (LLFF, N x 17)
    sparse/0/                  (COLMAP sparse model — for points3D if needed)
    hold=7                     (every 7th frame is test → exclude from train)

  /scratch-shared/qzhang1/scannetpp_processed/
    deblurnerf_pseudogt_evssm_como25k_bestiter/{scene}/{stem}.png
                               (sharp pseudo-GT at TRAIN poses, used as tgt)
    deblurnerf_pseudodepth_como25k/{scene}/{stem}.npy
                               (CoMoGaussian-rendered metric depth at TRAIN poses)

Training scheme (1-to-1 deblur, mirrors dataset_scannet_i2slam):
  context: N consecutive blurry frames from TRAIN cameras
  target : same N frames, image = pseudo-GT, depth = pseudo-depth
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
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DeblurNeRFBlurCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    pseudo_gt_root: Path
    pseudo_depth_root: Path

    near: float = 0.01
    far: float  = 100.0

    baseline_min: float = 0.0
    baseline_max: float = 1.0e10
    max_fov: float = 180.0
    make_baseline_1: bool = False
    augment: bool = False
    relative_pose: bool = False
    skip_bad_shape: bool = True

    train_scenes: list[str] = field(default_factory=lambda: [
        "blurball", "blurbasket", "blurbuick", "blurdecoration",
        "blurgirl", "blurheron", "blurparterre", "blurstair",
    ])
    val_scenes: list[str] = field(default_factory=lambda: [
        "blurcoffee", "blurpuppet",
    ])

    # Per-scene hold value is read from each scene dir's `hold=N` file.
    # If the file is missing, fall back to this default. (DeblurNeRF scenes
    # use mixed hold values — 6/7/8 — so a single global cfg.test_hold is wrong.)
    test_hold: int = 8
    samples_per_scene: int = 100
    val_samples_per_scene: int = 8
    num_context_views: int = 6
    frame_stride: int = 1


@dataclass
class DatasetDeblurNeRFBlurCfgWrapper:
    deblurnerf_blur_finetune: DeblurNeRFBlurCfg


def _load_poses_bounds(path: Path) -> tuple[np.ndarray, int, int, float]:
    data = np.load(path)
    poses = data[:, :15].reshape(-1, 3, 5)
    H = int(poses[0, 0, 4])
    W = int(poses[0, 1, 4])
    focal = float(poses[0, 2, 4])
    c2w_opencv = poses[:, :3, :4]
    N = c2w_opencv.shape[0]
    c2w = np.zeros((N, 4, 4), dtype=np.float32)
    c2w[:, :3, :4] = c2w_opencv
    c2w[:, 3, 3] = 1.0
    c2w[:, :3, 1:3] *= -1   # OpenCV → OpenGL
    return c2w, H, W, focal


class DatasetDeblurNeRFBlur(Dataset):
    cfg: DeblurNeRFBlurCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(self, cfg: DeblurNeRFBlurCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        scene_ids = cfg.train_scenes if stage == "train" else cfg.val_scenes

        self._scenes: dict[str, dict] = {}
        for s in scene_ids:
            for root in cfg.roots:
                root = Path(root)
                sd = root / s
                if not (sd / "poses_bounds.npy").exists():
                    continue
                c2w_all, H_full, W_full, focal_full = _load_poses_bounds(sd / "poses_bounds.npy")
                img_dir = sd / "images_4"
                img_files = sorted(img_dir.glob("*.png"))
                n = min(len(img_files), c2w_all.shape[0])
                if n == 0:
                    continue
                # Per-scene hold value (DeblurNeRF scenes use mixed 6/7/8).
                hold = cfg.test_hold
                for name in sd.iterdir():
                    if name.name.startswith("hold="):
                        try:
                            hold = int(name.name.split("=")[1])
                        except ValueError:
                            pass
                        break
                train_idxs = [i for i in range(n) if i % hold != 0]
                if len(train_idxs) == 0:
                    continue
                self._scenes[s] = {
                    "root": sd,
                    "c2w": c2w_all[:n].astype(np.float32),
                    "img_files": img_files[:n],
                    "stems": [p.stem for p in img_files[:n]],
                    "train_idxs": train_idxs,
                    "focal_full": focal_full,
                    "H_full": H_full,
                    "W_full": W_full,
                }
                break

        # Pre-compute sample index list for map-style access.
        # For val (deterministic): cycle through evenly-spaced starting positions.
        # For train: random starts each epoch (handled in __getitem__ via rng).
        n_per = cfg.samples_per_scene if stage == "train" else cfg.val_samples_per_scene
        N_ctx = cfg.num_context_views
        self._index: list[tuple[str, int]] = []   # (scene_id, start_in_train_idxs)
        for sid, sc in self._scenes.items():
            T = len(sc["train_idxs"])
            if T < N_ctx:
                continue
            max_start = T - N_ctx
            if stage == "train":
                # placeholder starts; randomized in __getitem__
                for i in range(n_per):
                    self._index.append((sid, -1))
            else:
                # deterministic, evenly-spaced starts
                if n_per == 1 or max_start == 0:
                    starts = [0]
                else:
                    step = max_start / max(n_per - 1, 1)
                    starts = [int(round(step * i)) for i in range(n_per)]
                for s_ in starts:
                    self._index.append((sid, s_))

    def __len__(self) -> int:
        return len(self._index)

    # ─────────────────────────────────────────────────────────────────────
    def _load_color(self, path: Path) -> Tensor:
        H_t, W_t = self.cfg.input_image_shape
        img = Image.open(path).convert("RGB").resize((W_t, H_t), Image.LANCZOS)
        return self.to_tensor(img)

    def _load_depth(self, path: Path) -> Tensor:
        H_t, W_t = self.cfg.input_image_shape
        d = np.load(path)
        if d.ndim == 3:
            d = d[0]
        d_t = torch.from_numpy(d).float()[None, None]
        d_t = F.interpolate(d_t, size=(H_t, W_t), mode="nearest")
        return d_t.squeeze(0).squeeze(0)

    def _intrinsics(self, scene: dict) -> Tensor:
        focal_4 = scene["focal_full"] / 4.0
        img_W_4, img_H_4 = Image.open(scene["img_files"][0]).size
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = focal_4 / img_W_4
        K[1, 1] = focal_4 / img_H_4
        K[0, 2] = 0.5
        K[1, 2] = 0.5
        return torch.from_numpy(K).float()

    def get_bound(self, bound: Literal["near", "far"], n: int) -> Float[Tensor, "v"]:
        return repeat(torch.tensor(getattr(self.cfg, bound), dtype=torch.float32), "-> v", v=n)

    # ─────────────────────────────────────────────────────────────────────
    def __getitem__(self, item: int):
        scene_id, start = self._index[item]
        scene = self._scenes[scene_id]
        train_idxs = scene["train_idxs"]
        N_ctx = self.cfg.num_context_views

        if start < 0:
            # train: randomize starting position
            start = random.randint(0, len(train_idxs) - N_ctx)
        idxs = [train_idxs[start + i] for i in range(N_ctx)]

        stems = [scene["stems"][i] for i in idxs]
        gt_dir = self.cfg.pseudo_gt_root / scene_id
        depth_dir = self.cfg.pseudo_depth_root / scene_id

        blur_imgs  = torch.stack([self._load_color(scene["img_files"][i]) for i in idxs])
        sharp_imgs = torch.stack([self._load_color(gt_dir / f"{stem}.png") for stem in stems])
        depths     = torch.stack([self._load_depth(depth_dir / f"{stem}.npy") for stem in stems])

        c2w = torch.from_numpy(scene["c2w"][idxs])
        K = self._intrinsics(scene).unsqueeze(0).expand(N_ctx, 3, 3).contiguous()
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
            "depth":       depths,
            "near":        self.get_bound("near", N_ctx),
            "far":         self.get_bound("far",  N_ctx),
            "index":       idx_t.clone(),
        }
        return {"context": ctx, "target": tgt, "scene": scene_id}

    @property
    def data_stage(self) -> Stage:
        if self.stage == "val":
            return "test"
        return self.stage
