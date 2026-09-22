"""Regression checks for the custom-dataset failures (run in the LTGS environment)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image
from src.utils.hloc_reference import validate_feature_tracks, validate_localization
from metrics import readImages, evaluate
from src.utils.camera_intrinsics import pinhole_intrinsics


class CustomDatasetTests(unittest.TestCase):
    def test_missing_object_descriptors_take_failed_match_path(self):
        from src.utils.registration_utils import find_3d_correspondences, run_teaserpp
        fields = dict(desc_3d_1=None, desc_3d_2=None, valid_idx_1=None,
                      valid_idx_2=None, proj1_xy=None, proj2_xy=None)
        before, after = find_3d_correspondences({}, {}, {1: fields}, [1], None)
        self.assertEqual(before[1].shape, (0, 3))
        self.assertEqual(after[1].shape, (0, 3))
        self.assertEqual(run_teaserpp(before, after, [1]), {})

    def test_independent_camera_intrinsics_preserve_projection(self):
        # Two phone views with different focal lengths, at portrait working size.
        expected_fx, expected_fy = np.array([400., 600.]), np.array([420., 630.])
        fov_x = 2 * np.arctan(288 / (2 * expected_fx))
        fov_y = 2 * np.arctan(512 / (2 * expected_fy))
        matrices = pinhole_intrinsics(fov_x, fov_y, 288, 512)
        projected = matrices @ np.array([1., 1., 2.])
        np.testing.assert_allclose(projected[:, :2] / projected[:, 2:],
                                   np.column_stack((expected_fx / 2 + 144, expected_fy / 2 + 256)))

    def test_degenerate_pose_with_many_inliers_is_rejected(self):
        model = SimpleNamespace(images={i: SimpleNamespace(projection_center=lambda i=i: np.array([i, 0., 0.])) for i in (0, 1)})
        ret = {'num_inliers': 55, 'inliers': [True] * 55,
               'cam_from_world': SimpleNamespace(inverse=lambda: SimpleNamespace(translation=np.array([1e14, 0., 0.])))}
        with self.assertRaisesRegex(ValueError, 'Implausible camera position'):
            validate_localization(ret, model, 'degenerate.jpeg')

    def test_weak_pose_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unreliable localization'):
            validate_localization({'num_inliers': 7, 'inliers': [False] * 202}, None, 'weak.jpeg')

    def test_tracks_require_matching_order_and_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'features.h5'
            coords = np.array([[1100, 20], [35, 48]], dtype=np.float16)
            with h5py.File(path, 'w') as handle:
                handle.create_dataset('frame.jpg/keypoints', data=coords)
            image = SimpleNamespace(name='frame.jpg', points2D=[SimpleNamespace(xy=p) for p in coords + .5])
            model = SimpleNamespace(images={1: image})
            validate_feature_tracks(model, path)
            image.points2D.reverse()
            with self.assertRaisesRegex(ValueError, 'Feature/track mismatch'):
                validate_feature_tracks(model, path)

    def test_empty_update_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in ('render_all', 'gt_all'):
                Path(directory, 'update', kind, '1').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'No evaluation pairs'):
                evaluate(directory)
            self.assertFalse(Path(directory, 'results.json').exists())

    def test_unpaired_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            render, gt = Path(directory, 'render'), Path(directory, 'gt')
            render.mkdir()
            gt.mkdir()
            Image.new('RGB', (8, 8)).save(render / 'a.png')
            Image.new('RGB', (8, 8)).save(gt / 'b.png')
            with self.assertRaisesRegex(ValueError, 'Unpaired images'):
                readImages(render, gt)


if __name__ == '__main__':
    unittest.main()
