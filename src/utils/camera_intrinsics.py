"""Pinhole calibration at the working image size, independently for each view."""
import numpy as np


def pinhole_intrinsics(fov_xs, fov_ys, width, height):
    fov_xs, fov_ys = np.asarray(fov_xs), np.asarray(fov_ys)
    if fov_xs.shape != fov_ys.shape or fov_xs.ndim != 1:
        raise ValueError('Provide one horizontal and vertical field of view per camera.')
    if width <= 0 or height <= 0 or not ((fov_xs > 0) & (fov_xs < np.pi) & (fov_ys > 0) & (fov_ys < np.pi)).all():
        raise ValueError('Invalid pinhole image dimensions or field of view.')
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (len(fov_xs), 1, 1))
    intrinsics[:, 0, 0] = width / (2 * np.tan(fov_xs / 2))
    intrinsics[:, 1, 1] = height / (2 * np.tan(fov_ys / 2))
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    return intrinsics
