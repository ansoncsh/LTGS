"""Keep feature indices and fixed-pose COLMAP tracks consistent."""
from pathlib import Path
import h5py
import numpy as np


def reference_model_path(hloc_dir):
    root = Path(hloc_dir)
    rebuilt = root / 'sparse_superpoint/0'
    return rebuilt if (rebuilt / 'images.bin').exists() else root / 'sparse/0'


def validate_feature_tracks(model, features):
    with h5py.File(features, 'r') as handle:
        for image in model.images.values():
            # HLoc adds 0.5 in the stored dtype (often float16).
            keypoints = handle[image.name]['keypoints'][:] + 0.5
            points = np.asarray([point.xy for point in image.points2D])
            if points.shape != keypoints.shape or not np.allclose(points, keypoints, atol=0.02):
                raise ValueError(f'Feature/track mismatch for {image.name}: '
                                 f'{len(keypoints)} features versus {len(points)} observations. '
                                 'Re-run preparation to triangulate SuperPoint tracks at fixed reference poses.')


def triangulate_reference(hloc_dir, images_dir):
    import pycolmap
    from hloc import triangulation
    root = Path(hloc_dir)
    output = root / 'sparse_superpoint/0'
    features = root / 'features.h5'
    if (output / 'images.bin').exists():
        model = pycolmap.Reconstruction(output)
    else:
        model = triangulation.main(
            output, root / 'sparse/0', Path(images_dir), root / 'pairs-netvlad.txt',
            features, root / 'matches.h5', mapper_options={
                'ba_refine_focal_length': False, 'ba_refine_principal_point': False,
                'ba_refine_extra_params': False})
    validate_feature_tracks(model, features)
    original = pycolmap.Reconstruction(root / 'sparse/0')
    if set(model.images) != set(original.images) or model.num_points3D() == 0:
        raise ValueError('Triangulation lost reference cameras or produced no points.')
    for image_id, image in model.images.items():
        if not np.allclose(image.cam_from_world.matrix(), original.images[image_id].cam_from_world.matrix(), atol=1e-8, rtol=0):
            raise ValueError('Reference poses changed; rebuilt tracks must share the Gaussian model coordinate frame.')
    for camera_id, camera in model.cameras.items():
        if not np.allclose(camera.params, original.cameras[camera_id].params, atol=1e-8, rtol=0):
            raise ValueError('Reference intrinsics changed during triangulation.')
    return output


def validate_localization(ret, model, name, min_inliers=20, min_inlier_ratio=0.05):
    count = 0 if ret is None else ret['num_inliers']
    ratio = 0 if ret is None else count / max(1, len(ret['inliers']))
    if count < min_inliers or ratio < min_inlier_ratio:
        raise ValueError(f'Unreliable localization for {name}: {count} inliers, {ratio:.1%} support '
                         f'(minimum {min_inliers}, {min_inlier_ratio:.1%}). Check feature tracks, '
                         'reference overlap and camera calibration before change detection.')
    centers = np.array([image.projection_center() for image in model.images.values()])
    center = ret['cam_from_world'].inverse().translation
    radius = np.linalg.norm(centers - np.median(centers, axis=0), axis=1).max()
    if not np.isfinite(center).all() or np.linalg.norm(center - np.median(centers, axis=0)) > 10 * max(radius, 1e-6):
        raise ValueError(f'Implausible camera position for {name}: outside ten times the reference camera extent.')
