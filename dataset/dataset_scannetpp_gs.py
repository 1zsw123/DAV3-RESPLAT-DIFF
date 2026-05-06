"""
Dataset loader — ScanNet++ GS pipeline with composite IQA blur detection.

Blur detection: ARNIQA-CSIQ >= 0.8 AND NIMA-KonIQ >= 0.7 → sharp
Pre-computed by scripts/precompute_iqa_scores.py → iqa_scores.json

Training scheme:
  Sharp frame  (composite sharp=True):
    context["image"] = iPhone frame
    target["image"]  = iPhone frame  (self-supervised)
    target["depth"]  = GS depth      (LiDAR depth TODO: replace with iPhone LiDAR)

  Blurry frame (composite sharp=False):
    context["image"] = iPhone frame
    target["image"]  = GS render     (pseudo-GT; TODO: enhance with video diffusion)
    target["depth"]  = GS depth

Dataset structure:
  root/{scene_id}/
    iphone_frames/    frame_XXXXXX.jpg
    gs_rgb_frames/    frame_XXXXXX.jpg   (GS render, exposure-compensated, sh=0)
    gs_depth_frames/  frame_XXXXXX.npy
    poses_gs.json
    iqa_scores.json   {fid: {arniqa, nima, sharp}}
"""

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
class ScannetppGsCfg(DatasetCfgCommon):
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
    num_context_views: int = 6
    min_sharp_frames: int = 20   # minimum sharp frames to include scene
    arniqa_threshold: float = 0.8
    nima_threshold: float   = 0.7


@dataclass
class DatasetScannetppGsCfgWrapper:
    scannetpp_gs: ScannetppGsCfg


class DatasetScannetppGs(IterableDataset):
    cfg: ScannetppGsCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.01
    far: float  = 10.0

    def __init__(self, cfg: ScannetppGsCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        if cfg.near > 0: self.near = cfg.near
        if cfg.far  > 0: self.far  = cfg.far

        self._scenes = list(cfg.scene_list) if cfg.scene_list else self._discover_scenes()

    def _is_sharp(self, iqa: dict) -> bool:
        return (iqa.get("arniqa", 0) >= self.cfg.arniqa_threshold and
                iqa.get("nima",   0) >= self.cfg.nima_threshold)

    def _discover_scenes(self) -> list[str]:
        scenes = []
        for root in self.cfg.roots:
            root = Path(root)
            for entry in sorted(root.iterdir()):
                if not entry.is_dir(): continue
                if not (entry / "poses_gs.json").exists(): continue
                if not (entry / "iqa_scores.json").exists(): continue
                if not (entry / "iphone_frames").exists(): continue
                try:
                    scores = json.load(open(entry / "iqa_scores.json"))
                    n_sharp = sum(1 for v in scores.values() if self._is_sharp(v))
                    if n_sharp < self.cfg.min_sharp_frames:
                        continue
                except Exception:
                    continue
                scenes.append(entry.name)
        return scenes

    def _load_image(self, path: Path) -> Float[Tensor, "3 h w"]:
        img = Image.open(path).convert("RGB")
        h, w = self.cfg.input_image_shape
        return self.to_tensor(img.resize((w, h), Image.LANCZOS))

    def _load_depth(self, path: Path) -> Float[Tensor, "h w"]:
        depth = np.load(str(path)).astype(np.float32)
        h, w = self.cfg.input_image_shape
        t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        return F.interpolate(t, size=(h, w), mode="nearest").squeeze(0).squeeze(0)

    def get_bound(self, bound: Literal["near", "far"], n: int) -> Float[Tensor, "view"]:
        return repeat(torch.tensor(getattr(self, bound), dtype=torch.float32), "-> v", v=n)

    def __iter__(self):
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
                print(f"[DatasetScannetppGs] Skipping {scene_name}: {e}")

    def _iter_scene(self, scene_name: str):
        scene_root = None
        for root in self.cfg.roots:
            c = Path(root) / scene_name
            if c.exists(): scene_root = c; break
        if scene_root is None:
            raise FileNotFoundError(scene_name)

        iphone_dir = scene_root / "iphone_frames"
        gs_rgb_dir = scene_root / "gs_rgb_frames"
        gs_dep_dir = scene_root / "gs_depth_frames"

        with open(scene_root / "poses_gs.json") as f:
            poses = json.load(f)
        with open(scene_root / "iqa_scores.json") as f:
            iqa = {int(k): v for k, v in json.load(f).items()}

        # annotate each pose entry
        valid = []
        for p in poses:
            fid = p["fid"]
            if not (iphone_dir / f"frame_{fid:06d}.jpg").exists():
                continue
            q = iqa.get(fid, {})
            valid.append({**p, "sharp": self._is_sharp(q),
                          "arniqa": q.get("arniqa", 0.0),
                          "nima":   q.get("nima",   0.0)})

        sharp_frames  = [p for p in valid if p["sharp"]]
        blurry_frames = [p for p in valid if not p["sharp"]]

        if len(sharp_frames) < self.cfg.num_context_views:
            raise ValueError(f"Too few sharp frames: {len(sharp_frames)}")

        n = self.cfg.num_context_views

        for _ in range(max(1, self.cfg.samples_per_scene)):
            # sample n_views — prefer sharp, mix in blurry if available
            sampled = random.sample(sharp_frames, n)

            try:
                iphone_imgs = torch.stack([
                    self._load_image(iphone_dir / f"frame_{p['fid']:06d}.jpg")
                    for p in sampled
                ])

                gt_imgs = []
                for p in sampled:
                    fid = p["fid"]
                    if p["sharp"]:
                        # self-supervised: GT = iPhone frame
                        gt_imgs.append(self._load_image(
                            iphone_dir / f"frame_{fid:06d}.jpg"))
                    else:
                        # blurry: GT = GS render (pseudo-GT)
                        gs_path = gs_rgb_dir / f"frame_{fid:06d}.jpg"
                        if gs_path.exists():
                            gt_imgs.append(self._load_image(gs_path))
                        else:
                            gt_imgs.append(self._load_image(
                                iphone_dir / f"frame_{fid:06d}.jpg"))
                gt_imgs = torch.stack(gt_imgs)

                depths = []
                for p in sampled:
                    dep = gs_dep_dir / f"frame_{p['fid']:06d}.npy"
                    if dep.exists():
                        depths.append(self._load_depth(dep))
                    else:
                        h, w = self.cfg.input_image_shape
                        depths.append(torch.zeros(h, w))
                depths = torch.stack(depths)

            except Exception:
                continue

            c2w = torch.tensor(np.array([p["c2w"] for p in sampled], np.float32))
            if self.cfg.relative_pose:
                c2w = camera_normalization(c2w[0:1], c2w)

            def make_K(d):
                return torch.tensor([[d["fx"],0,d["cx"]],[0,d["fy"],d["cy"]],[0,0,1]],
                                    dtype=torch.float32)
            K = torch.stack([make_K(p) for p in sampled])
            Kn = K.clone()
            Kn[:, 0, :] /= sampled[0]["w"]
            Kn[:, 1, :] /= sampled[0]["h"]

            sharp_mask = torch.tensor([p["sharp"] for p in sampled], dtype=torch.bool)
            fids       = torch.tensor([p["fid"]   for p in sampled], dtype=torch.long)

            yield {
                "context": {
                    "extrinsics": c2w,
                    "intrinsics": Kn,
                    "image":      iphone_imgs,
                    "near":       self.get_bound("near", n),
                    "far":        self.get_bound("far",  n),
                    "index":      fids,
                    "overlap":    torch.ones(n, dtype=torch.float32),
                },
                "target": {
                    "extrinsics": c2w,
                    "intrinsics": Kn,
                    "image":      gt_imgs,       # iPhone (sharp) or GS (blurry)
                    "depth":      depths,
                    "sharp_mask": sharp_mask,    # True=self-sup, False=GS pseudo-GT
                    "near":       self.get_bound("near", n),
                    "far":        self.get_bound("far",  n),
                    "index":      fids,
                },
                "scene": scene_name,
            }

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None: return "test"
        if self.stage == "val": return "test"
        return self.stage
