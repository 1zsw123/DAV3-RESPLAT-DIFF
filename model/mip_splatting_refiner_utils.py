"""
Shared geometry utilities for MipSplattingRefiner and DiffusionHead.
Covariance ↔ (log_scales, quaternions) conversions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def _cov_to_log_scales_quat(
    cov: Float[Tensor, "g 3 3"],
) -> tuple[Float[Tensor, "g 3"], Float[Tensor, "g 4"]]:
    """Decompose Σ = R diag(s²) Rᵀ → (log s, quat(R))."""
    eigenvalues, Q = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.clamp(min=1e-10)
    log_scales = eigenvalues.sqrt().log()

    dets = torch.linalg.det(Q)
    Q = Q.clone()
    Q[:, :, 0] = Q[:, :, 0] * dets.sign().unsqueeze(-1)

    quaternions = _matrix_to_quat(Q)
    return log_scales, quaternions


def _scales_quat_to_cov(
    scales: Float[Tensor, "g 3"],
    q: Float[Tensor, "g 4"],
) -> Float[Tensor, "g 3 3"]:
    """Build Σ = R diag(s²) Rᵀ from scales and unit quaternion."""
    R  = _quat_to_matrix(q)
    S2 = torch.diag_embed(scales ** 2)
    return R @ S2 @ R.transpose(-1, -2)


def _matrix_to_quat(R: Float[Tensor, "g 3 3"]) -> Float[Tensor, "g 4"]:
    """Rotation matrix → quaternion (w, x, y, z) via Shepperd's method."""
    s1 = torch.sqrt((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1.0).clamp(min=1e-10)) * 2
    s2 = torch.sqrt((1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    s3 = torch.sqrt((1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-10)) * 2
    s4 = torch.sqrt((1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-10)) * 2

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    w1 = 0.25 * s1;  x1 = (R[:, 2, 1] - R[:, 1, 2]) / s1
    y1 = (R[:, 0, 2] - R[:, 2, 0]) / s1;  z1 = (R[:, 1, 0] - R[:, 0, 1]) / s1

    w2 = (R[:, 2, 1] - R[:, 1, 2]) / s2;  x2 = 0.25 * s2
    y2 = (R[:, 0, 1] + R[:, 1, 0]) / s2;  z2 = (R[:, 0, 2] + R[:, 2, 0]) / s2

    w3 = (R[:, 0, 2] - R[:, 2, 0]) / s3;  x3 = (R[:, 0, 1] + R[:, 1, 0]) / s3
    y3 = 0.25 * s3;  z3 = (R[:, 1, 2] + R[:, 2, 1]) / s3

    w4 = (R[:, 1, 0] - R[:, 0, 1]) / s4;  x4 = (R[:, 0, 2] + R[:, 2, 0]) / s4
    y4 = (R[:, 1, 2] + R[:, 2, 1]) / s4;  z4 = 0.25 * s4

    cond1 = trace > 0
    cond2 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & ~cond1
    cond3 = (R[:, 1, 1] > R[:, 2, 2]) & ~cond1 & ~cond2

    w = torch.where(cond1, w1, torch.where(cond2, w2, torch.where(cond3, w3, w4)))
    x = torch.where(cond1, x1, torch.where(cond2, x2, torch.where(cond3, x3, x4)))
    y = torch.where(cond1, y1, torch.where(cond2, y2, torch.where(cond3, y3, y4)))
    z = torch.where(cond1, z1, torch.where(cond2, z2, torch.where(cond3, z3, z4)))

    return F.normalize(torch.stack([w, x, y, z], dim=-1), p=2, dim=-1)


def _quat_to_matrix(q: Float[Tensor, "g 4"]) -> Float[Tensor, "g 3 3"]:
    """Quaternion (w, x, y, z) → rotation matrix [G, 3, 3]."""
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    G = q.shape[0]

    R = torch.zeros(G, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z);  R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z);      R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y);      R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
