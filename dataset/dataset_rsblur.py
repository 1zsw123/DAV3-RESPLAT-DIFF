"""
Dataset loader for RSBlur real-motion-blur dataset.

Dataset structure:
  root/
  └── {scene_id}/               e.g. 0017
      └── {frame_id}/           e.g. 000339
          ├── real_blur/
          │   └── real_blur.png     blurry input (real long-exposure)
          └── gt/
              └── gt_sharp.png      sharp ground truth (beam-splitter ref)

No camera poses are provided. We use DA3's pose prediction (use_pred_pose=True
in EncoderDA3ReSplatCfg) — the dataset supplies identity poses as placeholders.

Training convention:
  - Context views : real_blur.png  (encoder input, V frames from the same scene)
  - Target views  : same indices   (deblurring task: target = sharp at same viewpoints)
  - Poses         : identity C2W   (DA3 predicts actual poses at runtime)
  - Loss          : image only (mse + lpips + ssim), no depth supervision
"""

from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf


def _stable_hash(s: str) -> int:
    """Process-stable hash, unlike Python's built-in hash() which is randomized
    per-process via PYTHONHASHSEED. Used to make val sample selection
    reproducible across training runs / DataLoader worker re-spawns so val/psnr
    numbers are comparable across runs (otherwise each run sees a different
    random subset of 200 val samples → ±2 PSNR noise even with same model)."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], 'big')
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class RSBlurCfg(DatasetCfgCommon):
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

    scene_list: list[str] = field(default_factory=list)
    val_scene_list: list[str] = field(default_factory=list)
    samples_per_scene: int = 200
    val_samples_per_scene: int = 50


@dataclass
class DatasetRSBlurCfgWrapper:
    rsblur: RSBlurCfg


class DatasetRSBlur(IterableDataset):
    cfg: RSBlurCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(self, cfg: RSBlurCfg, stage: Stage, view_sampler: ViewSampler) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self._scenes: list[tuple[str, list[Path]]] = self._discover_scenes()

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------

    def _discover_scenes(self) -> list[tuple[str, list[Path]]]:
        """Return list of (sub_scene_id, sorted frame dirs), split by stage.

        RSBlur scenes are NOT a single 3D scene — each scene_id contains many
        independent beam-splitter bursts, marked by gaps in frame_id numbering
        (consecutive frame_ids = same physical capture). Cross-burst sampling
        breaks the multi-view 3D assumption (DA3 pose pred, ReSplat triangulation).

        Fix: split each scene into burst sub-scenes by frame_id continuity.
        Each sub-scene name = "{scene_id}_b{burst_idx:02d}".
        Bursts smaller than MIN_BURST_FRAMES are dropped.

        If val_scene_list is set, match by SCENE PREFIX (before "_b") so
        e.g. "0016" matches "0016_b00", "0016_b01", ...
        """
        MIN_BURST_FRAMES = 16  # need >= num_context_views=16 (else sampler always raises)
        include = set(self.cfg.scene_list) if self.cfg.scene_list else None
        val_set = set(self.cfg.val_scene_list) if self.cfg.val_scene_list else None

        all_scenes = []
        for root in self.cfg.roots:
            root = Path(root)
            try:
                scene_names = sorted(os.listdir(root))
            except OSError:
                continue
            for scene_name in scene_names:
                if include and scene_name not in include:
                    continue
                scene_dir = root / scene_name
                try:
                    frame_names = sorted(
                        [n for n in os.listdir(scene_dir) if n.isdigit()]
                    )
                except OSError:
                    continue
                # Split into bursts by frame_id continuity
                bursts: list[list[Path]] = []
                cur: list[Path] = []
                prev_fid = None
                for fname in frame_names:
                    fid = int(fname)
                    if prev_fid is not None and fid - prev_fid > 1:
                        if len(cur) >= MIN_BURST_FRAMES:
                            bursts.append(cur)
                        cur = []
                    cur.append(scene_dir / fname)
                    prev_fid = fid
                if len(cur) >= MIN_BURST_FRAMES:
                    bursts.append(cur)
                # Each burst becomes its own sub-scene
                for bi, burst in enumerate(bursts):
                    sub_name = f"{scene_name}_b{bi:02d}"
                    all_scenes.append((sub_name, burst))

        # Match val by SCENE PREFIX (strip _b## suffix)
        def scene_prefix(sub_name: str) -> str:
            return sub_name.rsplit("_b", 1)[0]

        if val_set:
            if self.stage == "train":
                return [(n, f) for n, f in all_scenes if scene_prefix(n) not in val_set]
            else:
                return [(n, f) for n, f in all_scenes if scene_prefix(n) in val_set]
        return all_scenes

    # ------------------------------------------------------------------
    # Image loading (center-crop to target AR, then resize)
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> Float[Tensor, "3 h w"]:
        # Never throws — returns black placeholder if file missing/corrupt.
        # This is critical for DDP: all ranks must yield the same number of
        # samples per epoch or NCCL ALLREDUCE on val metrics times out.
        h_tgt, w_tgt = self.cfg.input_image_shape
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"[DatasetRSBlur] image load failed: {path} → black placeholder ({e})")
            return torch.zeros(3, h_tgt, w_tgt, dtype=torch.float32)
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

    def _load_mask(self, frame_dir: Path) -> Float[Tensor, "1 h w"]:
        # Returns [1, H, W] in [0,1]: 1=static (keep in loss), 0=dynamic (mask out).
        # Missing file or load failure → return all-ones (no masking).
        # Never throws — see _load_image note about DDP rank symmetry.
        h_tgt, w_tgt = self.cfg.input_image_shape
        p = frame_dir / "dynamic_mask.png"
        if not p.exists():
            return torch.ones(1, h_tgt, w_tgt, dtype=torch.float32)
        try:
            img = Image.open(p).convert("L")
        except (FileNotFoundError, OSError) as e:
            print(f"[DatasetRSBlur] mask load failed: {p} → all-ones placeholder ({e})")
            return torch.ones(1, h_tgt, w_tgt, dtype=torch.float32)
        orig_w, orig_h = img.size
        target_ar = w_tgt / h_tgt
        src_ar = orig_w / orig_h
        if src_ar > target_ar:
            new_w = int(orig_h * target_ar)
            left = (orig_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, orig_h))
        elif src_ar < target_ar:
            new_h = int(orig_w / target_ar)
            top = (orig_h - new_h) // 2
            img = img.crop((0, top, orig_w, top + new_h))
        img = img.resize((w_tgt, h_tgt), Image.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    # ------------------------------------------------------------------
    # Pose / intrinsics helpers (identity placeholders)
    # ------------------------------------------------------------------

    def _identity_c2w(self) -> np.ndarray:
        return np.eye(4, dtype=np.float32)

    def _load_scene_sharp_depths(self, scene_root: Path) -> dict[str, np.ndarray] | None:
        # Returns {frame_id_str: depth_HxW float16} from precomputed
        # da3_sharp_depths.npz (DA3 forward on gt_sharp.png). Used as
        # pseudo-GT depth for the depth loss.
        npz_path = scene_root / "da3_sharp_depths.npz"
        if not npz_path.exists():
            return None
        try:
            d = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"[DatasetRSBlur] failed to load {npz_path}: {e}")
            return None
        fids = d["frame_ids"]
        depths = d["depths"]  # [N, H, W] float16
        return {str(fids[i]): depths[i] for i in range(len(fids))}

    def _load_scene_poses(self, scene_root: Path) -> dict[str, tuple[int, np.ndarray]] | None:
        # Returns {frame_id_str: (chunk_id, c2w_4x4)} from precomputed da3_poses.npz.
        # None if missing — caller should fall back to no filtering.
        npz_path = scene_root / "da3_poses.npz"
        if not npz_path.exists():
            return None
        try:
            d = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"[DatasetRSBlur] failed to load {npz_path}: {e}")
            return None
        fids = d["frame_ids"]
        cids = d["chunk_ids"]
        poses = d["poses_c2w"]
        return {str(fids[i]): (int(cids[i]), poses[i]) for i in range(len(fids))}

    @staticmethod
    def _max_pairwise_rotation_deg(poses_c2w: np.ndarray) -> float:
        # Max pairwise rotation angle (deg) between cameras in poses_c2w [V, 4, 4].
        R = poses_c2w[:, :3, :3]
        V = R.shape[0]
        max_deg = 0.0
        for i in range(V):
            for j in range(i + 1, V):
                R_rel = R[i].T @ R[j]
                tr = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
                ang = float(np.degrees(np.arccos(tr)))
                if ang > max_deg:
                    max_deg = ang
        return max_deg

    def _default_intrinsics(self) -> np.ndarray:
        """Normalised K with fov≈60° as a reasonable default."""
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = 1.0    # fx/W  ≈ 1.0  (fov ≈ 53°)
        K[1, 1] = 1.0    # fy/H
        K[0, 2] = 0.5    # cx/W
        K[1, 2] = 0.5    # cy/H
        return K

    def get_bound(self, bound: Literal["near", "far"], n: int) -> Float[Tensor, "v"]:
        return repeat(torch.tensor(getattr(self.cfg, bound), dtype=torch.float32), "-> v", v=n)

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        # No DDP sharding here — matches the rest of the codebase (see
        # dataset_re10k.py).  Each rank iterates all scenes independently;
        # Lightning averages val metrics across ranks, which is correct when
        # all ranks see the same data.  Random view sampling gives each rank
        # different views during training.
        worker_info = torch.utils.data.get_worker_info()
        scenes = list(self._scenes)

        if worker_info is not None:
            n_w = worker_info.num_workers
            if len(scenes) >= n_w:
                scenes = [s for i, s in enumerate(scenes)
                          if i % n_w == worker_info.id]

        if self.stage == "train":
            perm = torch.randperm(len(scenes)).tolist()
            scenes = [scenes[i] for i in perm]

        for scene_name, frames in scenes:
            try:
                yield from self._iter_scene(scene_name, frames)
            except Exception as e:
                print(f"[DatasetRSBlur] Skipping scene {scene_name}: {e}")

    def _iter_scene(self, scene_name: str, frames: list[Path]):
        # CRITICAL: Use a per-(scene, sample_idx) deterministic seed for
        # view_sampler.sample() so ALL DDP ranks see the same ValueError
        # pattern → same yield count → no rank desync. Each rank still gets
        # variety across scenes (different perm in __iter__) and across
        # sample_idx, just consistently across ranks per scene.
        # Without this, each rank's RNG diverges → ValueError happens at
        # different indices → yield mismatch → DDP grad-reduce desync at
        # ~step 50 (NumelIn=1 vs NumelIn=param-bucket allreduce timeout).
        _rng_state = torch.get_rng_state()

        n = len(frames)
        K = self._default_intrinsics()

        # Build identity pose tensors for all frames
        c2w_all = torch.from_numpy(
            np.stack([self._identity_c2w()] * n, axis=0)
        )                                                   # [N, 4, 4]
        intr_all = torch.from_numpy(
            np.stack([K] * n, axis=0)
        )                                                   # [N, 3, 3]

        n_samples = (
            max(1, self.cfg.val_samples_per_scene)
            if self.stage != "train"
            else max(1, self.cfg.samples_per_scene)
        )

        # Pose-based filtering: VAL ONLY (train stays untouched for speed).
        # Reject ctx samples spanning multiple DA3 chunks or with max
        # pairwise rotation > VAL_MAX_ROT_DEG.
        VAL_MAX_ROT_DEG = 50.0
        VAL_FILTER_RETRIES = 16
        # Allow disabling val pose filter via env var for diagnostic A/B (test
        # whether pose filter biases val toward "stable burst" subset).
        scene_poses = (
            self._load_scene_poses(frames[0].parent)
            if self.stage != "train" and os.environ.get("DISABLE_POSE_FILTER", "0") != "1"
            else None
        )
        # Pseudo-GT depth from DA3 on gt_sharp.png (precomputed offline).
        # Loaded for both train AND val so depth loss can supervise.
        scene_sharp_depths = self._load_scene_sharp_depths(frames[0].parent)
        try:
            for _sample_idx in range(n_samples):
                context_indices = None
                overlap = None
                tries = (
                    VAL_FILTER_RETRIES
                    if (self.stage != "train" and scene_poses is not None)
                    else 1
                )
                for _t in range(tries):
                    # Deterministic seed across ranks: same scene + same
                    # sample_idx + same retry index → same sampler output.
                    torch.manual_seed(
                        (_stable_hash(scene_name) ^ (_sample_idx * 1000003) ^ (_t * 31337))
                        % (2 ** 31)
                    )
                    try:
                        ci, _, ov = self.view_sampler.sample(
                            scene_name, c2w_all, intr_all,
                        )
                    except ValueError:
                        ci = None
                        break
                    # Apply pose filter only when we have real DA3 poses (val).
                    # CHUNK_SIZE in precompute is 12, but we sample 16 ctx views,
                    # so requiring all-same-chunk would reject 100%. Instead:
                    # group ctx frames by chunk_id, check max rotation WITHIN
                    # each chunk; reject if any same-chunk pair > VAL_MAX_ROT_DEG.
                    # Frames in distinct chunks have incomparable poses → skipped
                    # (no info to filter on, accept by default).
                    if scene_poses is not None:
                        ctx_pose_lookup = [
                            scene_poses.get(frames[i].name) for i in ci.tolist()
                        ]
                        if any(p is None for p in ctx_pose_lookup):
                            continue  # frame has no precomputed pose; resample
                        by_chunk: dict[int, list[np.ndarray]] = {}
                        for cid, pose in ctx_pose_lookup:
                            by_chunk.setdefault(cid, []).append(pose)
                        rejected = False
                        for cid, poses_list in by_chunk.items():
                            if len(poses_list) < 2:
                                continue
                            poses_arr = np.stack(poses_list, axis=0)
                            if self._max_pairwise_rotation_deg(poses_arr) > VAL_MAX_ROT_DEG:
                                rejected = True
                                break
                        if rejected:
                            continue
                    context_indices = ci
                    overlap = ov
                    break
                if context_indices is None:
                    continue

                # CRITICAL: bounded view_sampler returns ctx as
                # [left, *random_extras, right] — NOT temporally sorted.
                # That made tgt = ctx[::2] arbitrary frames, which (a) made
                # val visualizations look misaligned (Context col vs GT col
                # rows are different physical frames), and (b) breaks the
                # implicit temporal-locality assumption used by DA3 pose
                # prediction. Sort once here so downstream sees a clean
                # temporally-ordered burst.
                context_indices = torch.sort(context_indices).values
                ctx_pos = context_indices.tolist()
                n_ctx = len(ctx_pos)
                # 1-to-1: tgt = ctx (was tgt = ctx[::2]). Each ctx blur frame
                # has its own paired sharp gt at the same frame_id (RSBlur
                # beam-splitter), so model is supervised on every ctx view.
                # Doubles render+loss compute but provides 2× supervision per
                # batch and removes Context/GT viz misalignment.
                tgt_pos = ctx_pos
                target_indices = context_indices

                # _load_image / _load_mask never throw (return placeholders on failure).
                blur_imgs = torch.stack([
                    self._load_image(frames[i] / "real_blur" / "real_blur.png")
                    for i in ctx_pos
                ])                                          # [V_ctx, 3, H, W]
                sharp_ctx_imgs = torch.stack([
                    self._load_image(frames[i] / "gt" / "gt_sharp.png")
                    for i in ctx_pos
                ])                                          # [V_ctx, 3, H, W]
                sharp_tgt_imgs = sharp_ctx_imgs              # [V_tgt=V_ctx, 3, H, W]
                mask_tgt = torch.stack([
                    self._load_mask(frames[i]) for i in tgt_pos
                ])                                          # [V_tgt, 1, H, W]

                ctx_ext  = c2w_all[context_indices]        # [V_ctx, 4, 4]
                ctx_intr = intr_all[context_indices]        # [V_ctx, 3, 3]
                tgt_ext  = ctx_ext.clone()                  # [V_tgt=V_ctx, 4, 4]
                tgt_intr = ctx_intr.clone()                 # [V_tgt=V_ctx, 3, 3]
                n_tgt = len(tgt_pos)

                ctx_dict = {
                    "extrinsics":  ctx_ext,
                    "intrinsics":  ctx_intr,
                    "image":       blur_imgs,
                    "sharp_image": sharp_ctx_imgs,
                    "near":  self.get_bound("near", n_ctx),
                    "far":   self.get_bound("far",  n_ctx),
                    "index": context_indices,
                    "overlap": overlap,
                }
                tgt_dict = {
                    "extrinsics": tgt_ext,
                    "intrinsics": tgt_intr,
                    "image":      sharp_tgt_imgs,
                    "mask":       mask_tgt,
                    "near":  self.get_bound("near", n_tgt),
                    "far":   self.get_bound("far",  n_tgt),
                    "index": target_indices,
                }
                # Pseudo-GT depth (DA3 on gt_sharp.png) for L_depth supervision.
                # If npz missing OR a tgt frame missing, omit → loss returns 0.
                if scene_sharp_depths is not None:
                    depth_lookup = [
                        scene_sharp_depths.get(frames[i].name) for i in tgt_pos
                    ]
                    if all(d is not None for d in depth_lookup):
                        tgt_dict["depth"] = torch.from_numpy(
                            np.stack(depth_lookup, axis=0).astype(np.float32)
                        )                                       # [V_tgt, H, W]

                yield {
                    "context": ctx_dict,
                    "target":  tgt_dict,
                    "scene":   scene_name,
                }
        finally:
            torch.set_rng_state(_rng_state)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage
