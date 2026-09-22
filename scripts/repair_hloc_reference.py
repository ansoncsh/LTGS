"""Rebuild SuperPoint tracks at fixed reference poses; test cached query matches.

Writes only to a new --output directory. Does not replace any pipeline artifacts.
Run from the repository root with its Python environment.
"""
import argparse
import json
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'submodules/Hierarchical-Localization'))
import h5py
import numpy as np
import pycolmap
from hloc import triangulation, extract_features, match_features
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils.io import find_pair, get_matches


def validate_feature_tracks(model, features):
    """HLoc uses feature indices as COLMAP point2D indices, not nearest pixels."""
    with h5py.File(features, 'r') as handle:
        for image in model.images.values():
            # Match HLoc's coordinate conversion, including stored float16 rounding.
            keypoints = handle[image.name]['keypoints'][:] + 0.5
            points = np.asarray([point.xy for point in image.points2D])
            if points.shape != keypoints.shape or not np.allclose(points, keypoints, atol=0.02):
                raise ValueError(f'Feature/track mismatch for {image.name}: '
                                 f'{len(keypoints)} features versus {len(points)} COLMAP observations. '
                                 'Triangulate the extracted features at the reference poses first.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hloc', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-results', type=Path, required=True)
    parser.add_argument('--reuse-reference', action='store_true')
    parser.add_argument('--max-references', type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.reuse_reference)
    base = args.hloc
    model = pycolmap.Reconstruction(args.output / 'reference') if args.reuse_reference else triangulation.main(
        args.output / 'reference', base / 'sparse/0', base.parent / 'images',
        base / 'pairs-netvlad.txt', base / 'features.h5', base / 'matches.h5',
        mapper_options={'ba_refine_focal_length': False, 'ba_refine_principal_point': False,
                        'ba_refine_extra_params': False})
    validate_feature_tracks(model, base / 'features.h5')
    original = pycolmap.Reconstruction(base / 'sparse/0')
    for image_id, image in model.images.items():
        if not np.allclose(image.cam_from_world.matrix(), original.images[image_id].cam_from_world.matrix(), atol=1e-8):
            raise ValueError('Triangulation changed reference poses; do not use with the existing Gaussian model.')
    old_results = json.loads(args.old_results.read_text())
    # The old test-localization path overwrote the query caches on some runs.
    # Build isolated, reusable caches and batch all query/reference pairs once.
    features_path = args.output / 'features.h5'
    matches_path = args.output / 'matches.h5'
    if not features_path.exists():
        shutil.copy2(base / 'features.h5', features_path)
    names = [entry['name'] for entry in old_results]
    extract_features.main(extract_features.confs['superpoint_aachen'], base.parent / 'images',
                          image_list=names, feature_path=features_path)
    pairs_path = args.output / 'pairs.txt'
    pairs_path.write_text(''.join(f'{name} {image.name}\n' for name in names for image in model.images.values()))
    match_features.main(match_features.confs['superglue'], pairs_path,
                        features=features_path, matches=matches_path)
    localizer = QueryLocalizer(model, {
        'estimation': {'ransac': {'max_error': 12, 'max_num_trials': 1000000, 'min_inlier_ratio': 0.01}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': False}})
    results, summary = [], []
    with h5py.File(matches_path, 'r') as matches:
        for old in old_results:
            name = old['name']
            inferred = pycolmap.infer_camera_from_image(base.parent / 'images' / name)
            camera = pycolmap.Camera(model='SIMPLE_PINHOLE', width=inferred.width,
                                    height=inferred.height, params=inferred.params[:3])
            ids = []
            for image_id, image in model.images.items():
                try:
                    find_pair(matches, name, image.name)
                    ids.append(image_id)
                except ValueError:
                    pass
            # Matching every reference introduces many contradictory 3D assignments.
            # Rank cached pairs by confident match support before PnP.
            ids = sorted(ids, key=lambda image_id: np.sum(get_matches(matches_path, name, model.images[image_id].name)[1] > 0.2), reverse=True)[:args.max_references]
            ret, log = pose_from_cluster(localizer, name, camera, ids,
                                         features_path, matches_path)
            count = 0 if ret is None else ret['num_inliers']
            total = 0 if ret is None else len(ret['inliers'])
            summary.append({'name': name, 'old_inliers': old['num_inliers'], 'inliers': count,
                            'correspondences': total, 'ratio': count / max(total, 1)})
            print(summary[-1], flush=True)
            if ret is not None:
                mask = np.asarray(ret['inliers'], dtype=bool)
                results.append(dict(name=name, model=camera.model.name, camera_id=camera.camera_id,
                                    width=camera.width, height=camera.height,
                                    rotation=ret['cam_from_world'].rotation.quat.tolist(),
                                    translation=ret['cam_from_world'].translation.tolist(),
                                    params=camera.params.tolist(), has_prior_focal_length=camera.has_prior_focal_length,
                                    num_inliers=count, inliers=mask.tolist(),
                                    points3D_ids=np.asarray(log['points3D_ids'])[mask].tolist(),
                                    xys=np.asarray(log['keypoints_query'])[mask].tolist()))
    (args.output / f'hloc_results_top{args.max_references}.json').write_text(json.dumps(results, indent=2))
    (args.output / f'localization_quality_top{args.max_references}.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
