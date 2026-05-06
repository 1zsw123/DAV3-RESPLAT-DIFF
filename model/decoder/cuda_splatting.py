from math import isqrt
from typing import Literal

import torch
from submodules.diff_gaussian_rasterization_w_pose.diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from submodules.diff_gaussian_rasterization_w_feature_detach.diff_gaussian_rasterization import (
    FeatureDetachGaussianRasterizationSettings,
    FeatureDetachGaussianRasterizer,
)

from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
    cx_norm: Float[Tensor, " batch"] | None = None,
    cy_norm: Float[Tensor, " batch"] | None = None,
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.

    cx_norm, cy_norm: per-batch normalized principal point (0.5 = centered).
    If given, builds an asymmetric frustum so that off-center principal points
    (cx_norm != 0.5, may even be < 0 or > 1 for out-of-image principals,
    e.g. multi-crop datasets) are correctly projected. If None, defaults to
    centered frustum (legacy behavior).
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    if cx_norm is None:
        right = tan_fov_x * near
        left = -right
    else:
        # Asymmetric frustum bounds at near plane such that principal axis
        # projects to pixel cx_norm * W (image space). Derivation:
        #   ndc_x = projmatrix[0,0] * X/Z + projmatrix[0,2]
        # Want pixel_x_norm = cx_norm + fx_norm * X/Z, so ndc_x = 2*pixel_x_norm - 1
        #   = 2*fx_norm * X/Z + (2*cx_norm - 1)
        # Match → projmatrix[0,2] = 2*cx_norm - 1, projmatrix[0,0] = 2*fx_norm
        # With fx_norm = 1/(2*tan_fov_x_centered):
        #   right - left = near/fx_norm = 2*tan_fov_x*near (same as centered)
        #   right + left = (2*cx_norm-1) * (right - left)
        right = 2 * cx_norm * tan_fov_x * near
        left = 2 * (cx_norm - 1) * tan_fov_x * near

    if cy_norm is None:
        top = tan_fov_y * near
        bottom = -top
    else:
        top = 2 * cy_norm * tan_fov_y * near
        bottom = 2 * (cy_norm - 1) * tan_fov_y * near

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_features: Float[Tensor, "batch gaussian feature_dim"] | None,
    scale_invariant: bool = True,
    use_sh: bool = True,
    cam_rot_delta: Float[Tensor, "batch 3"] | None = None,
    cam_trans_delta: Float[Tensor, "batch 3"] | None = None,
    low_pass_filter: float = 0.3,
    feature_detach: bool = False
) -> tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"], Float[Tensor, "batch out_feature_dim height width"] | None]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    # Compute CENTERED FOV directly from K's diagonal (not get_fov which gives
    # the asymmetric image-corner angle for off-center K). The rasterizer's
    # tanfovx/y is used for cov2D Jacobian where focal_x = (W/2)/tan_fov_x must
    # equal the true fx_pixel = fx_norm * W → tan_fov_x = 0.5 / fx_norm.
    fx_norm = intrinsics[:, 0, 0]
    fy_norm = intrinsics[:, 1, 1]
    cx_norm = intrinsics[:, 0, 2]
    cy_norm = intrinsics[:, 1, 2]
    tan_fov_x = 0.5 / fx_norm
    tan_fov_y = 0.5 / fy_norm
    fov_x = 2 * torch.atan(tan_fov_x)
    fov_y = 2 * torch.atan(tan_fov_y)

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y, cx_norm, cy_norm)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    all_depths = []
    all_features = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass
        
        if gaussian_features is None:    
            settings = GaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=tan_fov_x[i].item(),
                tanfovy=tan_fov_y[i].item(),
                bg=background_color[i],
                scale_modifier=1.0,
                viewmatrix=view_matrix[i],
                projmatrix=full_projection[i],
                projmatrix_raw=projection_matrix[i],
                sh_degree=degree,
                campos=extrinsics[i, :3, 3],
                prefiltered=False,  # This matches the original usage.
                debug=False,
                low_pass = low_pass_filter,
            )
            rasterizer = GaussianRasterizer(settings)

            row, col = torch.triu_indices(3, 3)


            image, radii, depth, opacity, n_touched = rasterizer(
                means3D=gaussian_means[i],
                means2D=mean_gradients,
                shs=shs[i] if use_sh else None,
                colors_precomp=None if use_sh else shs[i, :, 0, :],
                opacities=gaussian_opacities[i, ..., None],
                cov3D_precomp=gaussian_covariances[i, :, row, col],
                theta=cam_rot_delta[i] if cam_rot_delta is not None else None,
                rho=cam_trans_delta[i] if cam_trans_delta is not None else None,
            )
            all_images.append(image)
            all_radii.append(radii)
            all_depths.append(depth.squeeze(0))
            
            feature = None
        else:
            settings = FeatureDetachGaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=tan_fov_x[i].item(),
                tanfovy=tan_fov_y[i].item(),
                bg=background_color[i],
                scale_modifier=1.0,
                viewmatrix=view_matrix[i],
                projmatrix=full_projection[i],
                projmatrix_raw=projection_matrix[i],
                sh_degree=degree,
                campos=extrinsics[i, :3, 3],
                prefiltered=False,  # This matches the original usage.
                debug=False,
                low_pass = low_pass_filter,
            )
            rasterizer = FeatureDetachGaussianRasterizer(settings)

            row, col = torch.triu_indices(3, 3)

            image, features, radii, depth, opacity, n_touched = rasterizer(
                means3D=gaussian_means[i],
                means2D=mean_gradients,
                shs=shs[i] if use_sh else None,
                semantic_feature = gaussian_features[i],
                colors_precomp=None if use_sh else shs[i, :, 0, :],
                opacities=gaussian_opacities[i, ..., None],
                cov3D_precomp=gaussian_covariances[i, :, row, col],
                theta=cam_rot_delta[i] if cam_rot_delta is not None else None,
                rho=cam_trans_delta[i] if cam_trans_delta is not None else None,
            )
            all_images.append(image)
            all_radii.append(radii)
            all_depths.append(depth.squeeze(0))
            all_features.append(features)
            feature = torch.stack(all_features)
                        
    return torch.stack(all_images), torch.stack(all_depths), feature


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
    low_pass_filter = 0.3
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            projmatrix_raw=projection_matrix[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
            low_pass=low_pass_filter,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii, depth, opacity, n_touched = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]
