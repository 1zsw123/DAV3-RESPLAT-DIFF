"""Dataset loader for TUM RGB-D sequences (single-camera, real GT poses).

Used for DAV3 inference + ATE eval. Yields multi-view batches:
- context  = N consecutive TUM frames at their TUM groundtruth poses
- target   = same as context (deblur task: target view == input view)

The user's blur is whatever natural blur exists in the original TUM RGB
(no synthetic blur replacement). For ATE evaluation we save the rendered
deblurred image of EACH context view to disk via a model_wrapper hook,
then run Droid-SLAM on those PNGs.

Dataset structure expected:
  root/
  └── rgbd_dataset_freiburg{1,2,3}_<scene>/
      ├── rgb/<float-timestamp>.png          (640x480 RGB frames)
      └── groundtruth.txt                    (TUM format: ts tx ty tz qx qy qz qw)

Per-sequence intrinsics are baked in (TUM standard fr1/fr2/fr3 calibrations).
"""
from __future__ import annotations
import re
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


# TUM standard intrinsics (px, native 640×480)
TUM_CALIB = {
    "fr1": (517.3, 516.5, 318.6, 255.3),
    "fr2": (520.9, 521.0, 325.1, 249.7),
    "fr3": (535.4, 539.2, 320.1, 247.6),
}


@dataclass
class TumCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    near: float
    far: float

    # standard flags
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool

    # which sequences to use (e.g. ["rgbd_dataset_freiburg1_desk", ...])
    scene_list: list[str] = field(default_factory=list)

    samples_per_scene: int = 100
    output_image_shape: list[int] | None = None


@dataclass
class DatasetTumCfgWrapper:
    tum: TumCfg


def _calib_key_for_scene(scene_name: str) -> str:
    """`rgbd_dataset_freiburg1_desk` → `fr1`. Default fr3."""
    if "freiburg1" in scene_name: return "fr1"
    if "freiburg2" in scene_name: return "fr2"
    if "freiburg3" in scene_name: return "fr3"
    return "fr3"


def _quat_xyzw_to_R(q):
    qx, qy, qz, qw = q
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-12: return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz),  s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),      1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),      s*(qy*qz + qx*qw),     1 - s*(qx*qx + qy*qy)],
    ], dtype=np.float32)


def _load_tum_groundtruth(gt_path: Path) -> list[tuple[float, np.ndarray]]:
    """Parse TUM groundtruth.txt → list of (timestamp, c2w_4x4)."""
    rows = []
    for line in gt_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        parts = s.split()
        if len(parts) < 8: continue
        ts = float(parts[0])
        tx, ty, tz, qx, qy, qz, qw = (float(x) for x in parts[1:8])
        R = _quat_xyzw_to_R((qx, qy, qz, qw))
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = (tx, ty, tz)
        rows.append((ts, c2w))
    return rows


class DatasetTum(IterableDataset):
    cfg: TumCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.1
    far: float = 10.0

    def __init__(self, cfg: TumCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near > 0: self.near = cfg.near
        if cfg.far > 0: self.far = cfg.far

        if cfg.scene_list:
            self._scenes = cfg.scene_list
        else:
            self._scenes = self._discover_scenes()

    def _discover_scenes(self) -> list[str]:
        scenes = []
        for root in self.cfg.roots:
            root = Path(root)
            if not root.is_dir(): continue
            for entry in sorted(root.iterdir()):
                if not entry.is_dir(): continue
                if (entry / "rgb").is_dir() and (entry / "groundtruth.txt").exists():
                    scenes.append(entry.name)
        return scenes

    def _load_image(self, path: str) -> Float[Tensor, "3 h w"]:
        return self._load_image_at(path, self.cfg.input_image_shape)

    def _load_image_output(self, path: str) -> Float[Tensor, "3 h w"]:
        shape = self.cfg.output_image_shape or self.cfg.input_image_shape
        return self._load_image_at(path, shape)

    def _load_image_at(self, path: str, shape: list[int]) -> Float[Tensor, "3 h w"]:
        img = Image.open(path).convert("RGB")
        h, w = shape
        img = img.resize((w, h), Image.LANCZOS)
        return self.to_tensor(img)

    def _build_intrinsics(self, scene_name: str) -> np.ndarray:
        fx, fy, cx, cy = TUM_CALIB[_calib_key_for_scene(scene_name)]
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx / 640.; K[1, 1] = fy / 480.
        K[0, 2] = cx / 640.; K[1, 2] = cy / 480.
        return K

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Float[Tensor, "view"]:
        v = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(v, "-> v", v=num_views)

    def _scene_root(self, scene_name: str) -> Path:
        for r in self.cfg.roots:
            p = Path(r) / scene_name
            if p.exists(): return p
        raise FileNotFoundError(scene_name)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        scenes = list(self._scenes)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank(); ws = torch.distributed.get_world_size()
            scenes = [s for i, s in enumerate(scenes) if i % ws == rank]
        if worker_info is not None:
            scenes = [s for i, s in enumerate(scenes)
                      if i % worker_info.num_workers == worker_info.id]

        for scene_name in scenes:
            try:
                yield from self._iter_scene(scene_name)
            except Exception as e:
                print(f"[DatasetTum] Skipping scene {scene_name}: {e}")
                continue

    def _iter_scene(self, scene_name: str):
        scene_root = self._scene_root(scene_name)
        rgb_dir = scene_root / "rgb"
        gt_path = scene_root / "groundtruth.txt"

        # Sort RGB files by timestamp
        rgb_files = sorted([p for p in rgb_dir.iterdir() if p.suffix == ".png"],
                           key=lambda p: float(p.stem))
        if not rgb_files:
            raise ValueError(f"no rgb in {rgb_dir}")

        # Match each RGB ts to nearest GT pose
        gt = _load_tum_groundtruth(gt_path)
        if not gt: raise ValueError(f"no GT in {gt_path}")
        gt_ts = np.array([r[0] for r in gt])

        frame_data = []  # (seq_idx, rgb_path, c2w, ts)
        for seq_idx, rf in enumerate(rgb_files):
            ts = float(rf.stem)
            j = int(np.argmin(np.abs(gt_ts - ts)))
            if abs(gt_ts[j] - ts) > 0.05: continue
            frame_data.append((seq_idx, str(rf), gt[j][1], ts))

        if len(frame_data) < self.cfg.input_image_shape[0] + 4:
            raise ValueError(f"too few matched frames in {scene_name} ({len(frame_data)})")

        # Reindex sequentially after timestamp filtering
        for new_i, fd in enumerate(frame_data):
            frame_data[new_i] = (new_i, fd[1], fd[2], fd[3])
        n_frames = len(frame_data)

        # Persist seq_idx → original-TUM stem mapping so the post-processor
        # can rename the saved PNGs back to the original timestamp filenames.
        try:
            import json
            map_dir = Path("/scratch-shared/qzhang1/baselines/ate_eval/dav3_seq_maps")
            map_dir.mkdir(parents=True, exist_ok=True)
            mapping = {fd[0]: Path(fd[1]).stem for fd in frame_data}
            (map_dir / f"{scene_name}.json").write_text(
                json.dumps(mapping, indent=2))
        except Exception as _e:
            print(f"[DatasetTum] could not write seq map: {_e}")

        K = self._build_intrinsics(scene_name)
        intrinsics_all = torch.from_numpy(np.stack([K] * n_frames, axis=0))
        c2w_all = np.stack([fd[2] for fd in frame_data], axis=0)
        extrinsics_all = torch.from_numpy(c2w_all.astype(np.float32))

        n_samples = max(1, self.cfg.samples_per_scene)
        emitted: set[int] = set()
        for _ in range(n_samples):
            try:
                context_indices, _, overlap = self.view_sampler.sample(
                    scene_name, extrinsics_all, intrinsics_all,
                )
            except (ValueError, StopIteration):
                continue

            # Sequential sampler wraps once it exhausts the scene; stop the
            # moment we'd re-emit a chunk we've already produced.
            chunk_key = int(context_indices[0])
            if chunk_key in emitted:
                break
            emitted.add(chunk_key)

            target_indices = context_indices  # deblur: target = context
            context_pos = context_indices.tolist()
            target_pos  = target_indices.tolist()
            n_ctx = len(context_pos); n_tgt = len(target_pos)

            ctx_images = torch.stack([self._load_image(frame_data[i][1]) for i in context_pos])
            tgt_images = torch.stack([self._load_image_output(frame_data[i][1]) for i in target_pos])
            # No separate sharp GT for TUM — sharp_image == ctx_image
            ctx_sharp = torch.stack([self._load_image_output(frame_data[i][1]) for i in context_pos])

            ctx_extrinsics = extrinsics_all[context_indices]
            tgt_extrinsics = extrinsics_all[target_indices]
            ctx_intrinsics = intrinsics_all[context_indices]
            tgt_intrinsics = intrinsics_all[target_indices]

            all_sel_ext = torch.cat([ctx_extrinsics, tgt_extrinsics], dim=0)
            scale = 1.0
            if self.cfg.make_baseline_1:
                a = ctx_extrinsics[0, :3, 3]; b = ctx_extrinsics[-1, :3, 3]
                scale = float((a - b).norm())
                if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                    continue
                all_sel_ext = all_sel_ext.clone()
                all_sel_ext[:, :3, 3] /= scale

            if self.cfg.relative_pose:
                all_sel_ext = camera_normalization(all_sel_ext[0:1], all_sel_ext)

            ctx_extrinsics = all_sel_ext[:n_ctx]
            tgt_extrinsics = all_sel_ext[n_ctx:]

            # Frame stems for output naming (preserve original TUM PNG filenames)
            ctx_stems = [Path(frame_data[i][1]).stem for i in context_pos]

            target_dict = {
                "extrinsics": tgt_extrinsics,
                "intrinsics": tgt_intrinsics,
                "image":      tgt_images,
                "near":       self.get_bound("near", n_tgt) / scale,
                "far":        self.get_bound("far",  n_tgt) / scale,
                "index":      target_indices,
            }
            ctx_dict = {
                "extrinsics":   ctx_extrinsics,
                "intrinsics":   ctx_intrinsics,
                "image":        ctx_images,
                "sharp_image":  ctx_sharp,
                "near":         self.get_bound("near", n_ctx) / scale,
                "far":          self.get_bound("far",  n_ctx) / scale,
                "index":        context_indices,
                "overlap":      overlap,
            }
            yield {
                "context": ctx_dict,
                "target":  target_dict,
                "scene":   scene_name,
                "ctx_stems": ctx_stems,  # for saving deblurred outputs by original filename
            }

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None: return "test"
        if self.stage == "val": return "test"
        return self.stage
