"""Sequential view sampler: deterministic non-overlapping chunks of N frames.

Yields chunk i = [i*N, i*N+1, ..., i*N+N-1] for the i-th call within a scene.
Across all chunks, every frame in the scene appears exactly once as both
context and target (since `dataset_blur_replica` overrides target_indices =
context_indices).

Use case: stress-test that bounded-sampler val numbers reflect real model
performance, not selection bias from random 45-90 frame gaps.
"""
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerSequentialCfg:
    name: Literal["sequential"]
    num_context_views: int  # frames per chunk
    num_target_views: int   # ignored — dataset overrides target = context
    # samples_per_scene in dataset cfg controls how many chunks are yielded.


class ViewSamplerSequential(ViewSampler[ViewSamplerSequentialCfg]):
    """Stateful: each call returns the next sequential chunk; resets when scene changes."""

    def __init__(self, cfg, stage, overfit, cameras_are_circular, step_tracker):
        super().__init__(cfg, stage, overfit, cameras_are_circular, step_tracker)
        self._scene_chunk_counter: dict[str, int] = {}

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
        Float[Tensor, ""],
    ]:
        v = extrinsics.shape[0]
        n = self.cfg.num_context_views
        chunk_idx = self._scene_chunk_counter.get(scene, 0)
        start = (chunk_idx * n) % max(1, v - n + 1)  # wrap so we don't go OOB on last partial chunk
        end = start + n
        if end > v:
            # last chunk: pad by repeating last frame so we still get N indices
            ctx = list(range(v - n, v))
        else:
            ctx = list(range(start, end))
        self._scene_chunk_counter[scene] = chunk_idx + 1

        index_context = torch.tensor(ctx, device=device, dtype=torch.int64)
        # Target is overridden by dataset_blur_replica anyway; return same indices.
        index_target = index_context.clone()
        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)
        return index_context, index_target, overlap

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
