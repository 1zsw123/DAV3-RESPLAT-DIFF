"""
Dataset loader for blur/sharp Replica sequences.

Dataset structure expected:
  root/
  └── {scene}/               e.g. office_3
      ├── traj_8.txt         camera-to-world poses, one 4x4 matrix per line (16 values)
      └── {blur_dir}/        e.g. blur_36
          ├── rgb_XXXXX.png  blurry input frames
          └── results/
              └── rgb/
                  └── rgb_XXXXX.png  corresponding sharp GT frames

Training convention:
  - Context views  : blurry frames  (encoder input)
  - Target views   : sharp frames   (rendering GT)
  - Poses are shared (blur and sharp are captured at the same camera positions)
"""

import os
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


@dataclass
class BlurReplicaCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]

    # --- blur/sharp directory names ---
    blur_dir: str           # sub-dir containing blurry frames, e.g. "blur_36"
    traj_file: str          # trajectory filename, e.g. "traj_8.txt"

    # --- camera intrinsics (pixels, before normalisation) ---
    fx: float               # focal length x
    fy: float               # focal length y
    cx: float               # principal point x
    cy: float               # principal point y
    original_width: int     # width of raw frames
    original_height: int    # height of raw frames

    # --- depth / scene bounds ---
    near: float
    far: float

    # --- standard flags shared with other datasets ---
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool

    # --- optional: list of scene sub-directories to use (None = auto-discover) ---
    scene_list: list[str] = field(default_factory=list)

    # --- baseline mode: use sharp frames as context (True) or blur frames (False) ---
    sharp_context: bool = True

    # --- how many times to sample from each scene per epoch ---
    samples_per_scene: int = 100

    # --- GT depth maps directory name (relative to scene root); None = no depth ---
    # Depth files: depth/depth_{frame_idx:05d}.npy, float32, metres, 0=invalid
    depth_dir: str = "depth"

    # --- output resolution for target (sharp GT) images and decoder rendering ---
    # If None, falls back to input_image_shape (old behaviour).
    # Set to [300, 600] to supervise loss at higher resolution than encoder input.
    output_image_shape: list[int] | None = None


@dataclass
class DatasetBlurReplicaCfgWrapper:
    blur_replica: BlurReplicaCfg


# ---------------------------------------------------------------------------
# Helper: parse trajectory file
# ---------------------------------------------------------------------------

def _load_trajectory(traj_path: str) -> dict[int, np.ndarray]:
    """
    Read a trajectory file and return a dict {frame_index: 4x4_C2W_matrix}.

    Supported formats
    -----------------
    1. One frame per line, 16 space-separated floats (row-major 4x4 C2W).
       Line index == frame index.
    2. Blocks of 4 lines (each with 4 floats) separated by blank lines.
       Block index == frame index.
    """
    with open(traj_path, "r") as f:
        raw = f.read()

    # Split into non-empty lines
    lines = [l.strip() for l in raw.splitlines()]
    non_empty = [l for l in lines if l and not l.startswith("#")]

    if not non_empty:
        raise ValueError(f"Empty trajectory file: {traj_path}")

    poses: dict[int, np.ndarray] = {}
    first_tokens = non_empty[0].split()

    if len(first_tokens) == 16:
        # Format 1: 16 values per line
        for idx, line in enumerate(non_empty):
            vals = list(map(float, line.split()))
            poses[idx] = np.array(vals, dtype=np.float32).reshape(4, 4)

    elif len(first_tokens) == 4:
        # Format 2: 4 lines per frame, optional blank-line separator
        # Collect blocks by splitting on blank lines first
        all_lines = raw.splitlines()
        blocks, block = [], []
        for l in all_lines:
            stripped = l.strip()
            if stripped.startswith("#"):
                continue
            if stripped == "":
                if block:
                    blocks.append(block)
                    block = []
            else:
                block.append(stripped)
        if block:
            blocks.append(block)

        frame_idx = 0
        for blk in blocks:
            rows = []
            for l in blk:
                vals = list(map(float, l.split()))
                rows.append(vals)
                if len(rows) == 4:
                    poses[frame_idx] = np.array(rows, dtype=np.float32)
                    rows = []
                    frame_idx += 1

    else:
        raise ValueError(
            f"Unrecognised trajectory format in {traj_path}: "
            f"first line has {len(first_tokens)} tokens (expected 4 or 16)."
        )

    return poses


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetBlurReplica(IterableDataset):
    cfg: BlurReplicaCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.01
    far: float = 100.0

    def __init__(
        self,
        cfg: BlurReplicaCfg,
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

        # Discover scenes
        if cfg.scene_list:
            self._scenes = cfg.scene_list
        else:
            self._scenes = self._discover_scenes()

        # ── Pre-load canonical GT Gaussians (means in world frame) per scene ──
        # Generated by scripts/precompute_sharp_gaussians.py (Option C).
        # Dict: scene_name → Float32 tensor [G, 3]
        self._canonical_gs: dict[str, torch.Tensor] = {}
        for scene_name in self._scenes:
            for root in self.cfg.roots:
                gt_path = Path(root) / scene_name / "sharp_gaussians" / "canonical_gs.pt"
                if gt_path.exists():
                    data = torch.load(gt_path, map_location="cpu", weights_only=False)
                    # fp16 on disk → fp32 in memory
                    self._canonical_gs[scene_name] = data["means_world"].float()
                    break

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------

    def _discover_scenes(self) -> list[str]:
        scenes = []
        for root in self.cfg.roots:
            root = Path(root)
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                traj = entry / self.cfg.traj_file
                blur = entry / self.cfg.blur_dir
                if traj.exists() and blur.exists():
                    scenes.append(entry.name)
        return scenes

    # ------------------------------------------------------------------
    # Intrinsics (normalised 3×3 K)
    # ------------------------------------------------------------------

    def _build_intrinsics(self) -> np.ndarray:
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self.cfg.fx / self.cfg.original_width
        K[1, 1] = self.cfg.fy / self.cfg.original_height
        K[0, 2] = self.cfg.cx / self.cfg.original_width
        K[1, 2] = self.cfg.cy / self.cfg.original_height
        return K

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_index_from_name(filename: str) -> int:
        """Extract integer index from filenames like 'rgb_10008.png'."""
        m = re.search(r"(\d+)", Path(filename).stem)
        if m is None:
            raise ValueError(f"Cannot parse frame index from: {filename}")
        return int(m.group(1))

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

    def _load_depth(self, depth_dir: Path, frame_idx: int) -> Float[Tensor, "h w"] | None:
        """Load pre-computed GT depth (float32 .npy, metres, 0=invalid).
        Returns None if the file doesn't exist yet (depth generation still running).
        Resizes to output_image_shape so it matches the rendered depth resolution.
        """
        path = depth_dir / f"depth_{frame_idx:05d}.npy"
        if not path.exists():
            return None
        depth = torch.from_numpy(np.load(str(path)).astype(np.float32))
        out_h, out_w = (self.cfg.output_image_shape or self.cfg.input_image_shape)
        if depth.shape != (out_h, out_w):
            import torch.nn.functional as F
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(out_h, out_w), mode="nearest"
            ).squeeze(0).squeeze(0)
        return depth

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    def _make_extrinsics(self, c2w: np.ndarray) -> np.ndarray:
        """Convert C2W (4x4) → W2C (4x4) as expected by C3G."""
        return np.linalg.inv(c2w).astype(np.float32)

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Float[Tensor, "view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        scenes = self._scenes

        # Shard scenes across DDP ranks (training and val/test)
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

        # Shuffle scenes during training
        if self.stage == "train":
            perm = torch.randperm(len(scenes)).tolist()
            scenes = [scenes[i] for i in perm]

        for scene_name in scenes:
            try:
                yield from self._iter_scene(scene_name)
            except Exception as e:
                print(f"[DatasetBlurReplica] Skipping scene {scene_name}: {e}")
                continue

    def _iter_scene(self, scene_name: str):
        # Locate directories
        scene_root = None
        for root in self.cfg.roots:
            candidate = Path(root) / scene_name
            if candidate.exists():
                scene_root = candidate
                break
        if scene_root is None:
            raise FileNotFoundError(f"Scene not found: {scene_name}")

        blur_dir   = scene_root / self.cfg.blur_dir
        sharp_dir  = scene_root / "results" / "rgb"   # sharp 帧在 scene_root/results/rgb/
        depth_dir  = scene_root / self.cfg.depth_dir  # GT depth maps (may not exist yet)

        # ── Blur window half-offset ──────────────────────────────────────────
        # blur_36/rgb_i.png integrates sharp frames [i, i+35].
        # The true corresponding sharp frame is the MIDPOINT: rgb_{i + blur_half}.png
        # where blur_half = 36 // 2 = 18.
        try:
            blur_window = int(self.cfg.blur_dir.split("_")[-1])
        except (ValueError, IndexError):
            blur_window = 0
        blur_half = blur_window // 2

        # ── Use mid_traj.txt if available (sequential, one entry per blur frame)
        # mid_traj.txt line j == C2W pose for blur frame j (= midpoint of its window).
        # Fall back to the configured traj_file if mid_traj.txt doesn't exist.
        mid_traj_path = blur_dir / "mid_traj.txt"
        if mid_traj_path.exists():
            mid_poses_raw = _load_trajectory(str(mid_traj_path))
            # mid_traj is sequentially indexed (0, 1, 2, ...) — map seq_idx → pose
            mid_poses = mid_poses_raw   # dict {seq_idx: 4x4}
            use_mid_traj = True
        else:
            # Fall back: load full traj and look up by midpoint frame number
            traj_path = scene_root / self.cfg.traj_file
            mid_poses = _load_trajectory(str(traj_path))
            use_mid_traj = False

        # Collect blurry frames that also have a sharp counterpart and a pose
        # frame_data: list of (seq_idx, blur_path, sharp_path, c2w, fidx_for_depth)
        # Sort numerically by frame index so seq_idx matches mid_traj.txt ordering.
        blur_files = sorted(
            blur_dir.glob("rgb_*.png"),
            key=lambda p: self._frame_index_from_name(p.name),
        )
        frame_data  = []

        for seq_idx, bf in enumerate(blur_files):
            fidx = self._frame_index_from_name(bf.name)   # e.g. 0, 36, 72 ...

            # Corresponding sharp frame: midpoint of blur window
            sharp_fidx = fidx + blur_half
            sf = sharp_dir / f"rgb_{sharp_fidx:05d}.png"
            if not sf.exists():
                # Try without zero-padding (some datasets use rgb_18.png)
                sf = sharp_dir / f"rgb_{sharp_fidx}.png"
            if not sf.exists():
                continue

            # Pose for this blur frame (midpoint camera)
            if use_mid_traj:
                pose_key = seq_idx
            else:
                pose_key = sharp_fidx
            if pose_key not in mid_poses:
                continue

            # Tuple: (seq_idx, blur_path, sharp_path, c2w, fidx_for_depth)
            # fidx is the original blur-frame number (0, 36, 72...) used for depth filenames.
            frame_data.append((seq_idx, str(bf), str(sf), mid_poses[pose_key], fidx))

        if len(frame_data) < 3:
            raise ValueError(f"Not enough valid frames in {scene_name} (found {len(frame_data)})")

        frame_data.sort(key=lambda x: x[0])
        n_frames = len(frame_data)

        # ---- build pose/intrinsics tensors for ALL frames ----
        K = self._build_intrinsics()
        intrinsics_all = torch.from_numpy(np.stack([K] * n_frames, axis=0))
        c2w_all = np.stack([fd[3] for fd in frame_data], axis=0)
        extrinsics_all = torch.from_numpy(c2w_all.astype(np.float32))

        ctx_src = 2 if self.cfg.sharp_context else 1  # 2=sharp, 1=blur

        # ---- sample multiple examples from this scene, just like RE10K ----
        n_samples = max(1, self.cfg.samples_per_scene)
        for _ in range(n_samples):
            try:
                context_indices, _, overlap = self.view_sampler.sample(
                    scene_name,
                    extrinsics_all,
                    intrinsics_all,
                )
            except ValueError:
                continue

            # Deblurring task: target = same viewpoints as context.
            # Input is blurry context; GT is the sharp image at the SAME camera pose.
            target_indices = context_indices

            context_pos = context_indices.tolist()
            target_pos  = target_indices.tolist()
            n_ctx = len(context_pos)
            n_tgt = len(target_pos)

            # ---- load images ----
            ctx_images = torch.stack([
                self._load_image(frame_data[i][ctx_src]) for i in context_pos
            ])
            tgt_images = torch.stack([
                self._load_image_output(frame_data[i][2]) for i in target_pos  # sharp GT at context poses
            ])

            # ---- sharp context images (always loaded for L_flow supervision) ----
            # frame_data[i][2] is the sharp image path (index 2 = sharp_path)
            ctx_sharp_images = torch.stack([
                self._load_image_output(frame_data[i][2]) for i in context_pos
            ])  # [V_ctx, 3, H_out, W_out]  sharp frames at context indices

            # ---- load GT depth maps (optional, None if not yet generated) ----
            has_depth = depth_dir.exists()
            tgt_depths = None
            ctx_depths = None
            if has_depth:
                depth_list = [
                    self._load_depth(depth_dir, frame_data[i][4]) for i in target_pos
                ]
                # Only use depth if ALL target frames have a depth file
                if all(d is not None for d in depth_list):
                    tgt_depths = torch.stack(depth_list)   # [V_tgt, H, W]

                ctx_depth_list = [
                    self._load_depth(depth_dir, frame_data[i][4]) for i in context_pos
                ]
                if all(d is not None for d in ctx_depth_list):
                    ctx_depths = torch.stack(ctx_depth_list)  # [V_ctx, H, W]

            # ---- build pose tensors ----
            ctx_extrinsics = extrinsics_all[context_indices]
            tgt_extrinsics = extrinsics_all[target_indices]
            ctx_intrinsics = intrinsics_all[context_indices]
            tgt_intrinsics = intrinsics_all[target_indices]

            all_sel_ext = torch.cat([ctx_extrinsics, tgt_extrinsics], dim=0)

            # ---- baseline scaling ----
            scale = 1.0
            if self.cfg.make_baseline_1:
                a = ctx_extrinsics[0,  :3, 3]
                b = ctx_extrinsics[-1, :3, 3]
                scale = float((a - b).norm())
                if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                    continue
                all_sel_ext = all_sel_ext.clone()
                all_sel_ext[:, :3, 3] /= scale

            # ---- relative pose normalisation ----
            if self.cfg.relative_pose:
                all_sel_ext = camera_normalization(all_sel_ext[0:1], all_sel_ext)

            ctx_extrinsics = all_sel_ext[:n_ctx]
            tgt_extrinsics = all_sel_ext[n_ctx:]

            target_dict = {
                "extrinsics": tgt_extrinsics,
                "intrinsics": tgt_intrinsics,
                "image":      tgt_images,
                "near":   self.get_bound("near", n_tgt) / scale,
                "far":    self.get_bound("far",  n_tgt) / scale,
                "index":  target_indices,
            }
            if tgt_depths is not None:
                target_dict["depth"] = tgt_depths   # [V_tgt, H, W] float32 metres

            ctx_dict = {
                "extrinsics":   ctx_extrinsics,
                "intrinsics":   ctx_intrinsics,
                "image":        ctx_images,
                "sharp_image":  ctx_sharp_images,  # sharp frames at context indices
                "near":   self.get_bound("near", n_ctx) / scale,
                "far":    self.get_bound("far",  n_ctx) / scale,
                "index":  context_indices,
                "overlap": overlap,
            }
            if ctx_depths is not None:
                ctx_dict["depth"] = ctx_depths  # [V_ctx, H, W] GT depth at context poses

            batch = {
                "context": ctx_dict,
                "target":  target_dict,
                "scene":   scene_name,
            }
            # Canonical GT sharp Gaussians in world frame [G, 3], if precomputed
            if scene_name in self._canonical_gs:
                batch["gt_means_world"] = self._canonical_gs[scene_name]

            yield batch

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage
