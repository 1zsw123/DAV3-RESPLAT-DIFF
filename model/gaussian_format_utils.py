# gaussian_format_utils.py - format conversions between DA3/ReSplat/C3G Gaussians
# See docstring in each function for details.
from __future__ import annotations
import sys
from pathlib import Path
import torch, torch.nn.functional as F
from torch import Tensor

_RESPLAT_ROOT = str(Path(__file__).resolve().parent.parent.parent / "resplat")

def _ensure_resplat_path():
    if _RESPLAT_ROOT not in sys.path:
        sys.path.insert(0, _RESPLAT_ROOT)

def quat_wxyz_to_matrix(q):
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]
    R = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], dim=-1).reshape(*B, 3, 3)
    return R

def build_cov(scales, quats_wxyz):
    R = quat_wxyz_to_matrix(quats_wxyz); S2 = torch.diag_embed(scales**2)
    return R @ S2 @ R.transpose(-1,-2)

def dav3_to_resplat_gaussians(dav3_gs, sh_target_d=9):
    _ensure_resplat_path()
    from src.model.types import Gaussians as RS
    means=dav3_gs.means; scales=dav3_gs.scales; rotations=dav3_gs.rotations
    harmonics=dav3_gs.harmonics; opacities=dav3_gs.opacities
    if opacities.dim()==4: opacities=opacities[:,:,0,0]
    d=harmonics.shape[-1]
    if d<sh_target_d:
        pad=torch.zeros(*harmonics.shape[:-1],sh_target_d-d,device=harmonics.device,dtype=harmonics.dtype)
        harmonics=torch.cat([harmonics,pad],dim=-1)
    elif d>sh_target_d: harmonics=harmonics[...,:sh_target_d]
    cov=build_cov(scales,rotations)
    return RS(means=means,covariances=cov,harmonics=harmonics,opacities=opacities,
              scales=scales,rotations=rotations,rotations_unnorm=rotations.clone())

def resplat_to_c3g_gaussians(rs_gs):
    from .types import Gaussians as C3G
    return C3G(means=rs_gs.means,covariances=rs_gs.covariances,harmonics=rs_gs.harmonics,opacities=rs_gs.opacities,feature=None)

def c2w_to_da3_w2c_norm(ctx_c2w):
    ctx_w2c=torch.inverse(ctx_c2w); transform=torch.inverse(ctx_w2c[0:1])
    w2c_norm=ctx_w2c@transform.expand(len(ctx_w2c),-1,-1)
    c2w_n=torch.inverse(w2c_norm); dists=c2w_n[:,:3,3].norm(dim=-1)
    md=torch.median(dists).clamp(min=0.1); w2c_norm[:,:3,3]/=md
    return w2c_norm, transform, md.item()
