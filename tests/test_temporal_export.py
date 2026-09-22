import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gaussian-splatting'))
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render, dynamic_render


class TemporalExportTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA rasterizer required')
    def test_snapshot_matches_dynamic_render_with_pose_correction(self):
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            model = GaussianModel(3)
            model._xyz = torch.tensor([[0., 0., 3.], [1., 0., 3.]], device='cuda')
            model._rotation = torch.tensor([[1., 0., 0., 0.]] * 2, device='cuda')
            model._scaling = torch.full((2, 3), -2., device='cuda')
            model._opacity = torch.ones((2, 1), device='cuda')
            model._features_dc = torch.ones((2, 1, 3), device='cuda')
            model._features_rest = torch.zeros((2, 15, 3), device='cuda')
            model.active_sh_degree = 3
            camera = Camera((32, 32), 0, np.eye(3), np.zeros(3), 1., 1., None,
                            Image.new('RGB', (32, 32)), None, 'test', 0)
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                                   debug=False, antialiasing=False)
            indices, tracks = {1: np.array([0]), 2: np.array([1])}, {1: [0, 1], 2: [0]}
            transforms = {1: {1: np.eye(4)}}
            masks = {1: torch.tensor([[1.], [0.]], device='cuda')}
            errors = {1: {1: torch.tensor([.3, 0., 0., 0., .2, 0.], device='cuda')}}
            background = torch.zeros(3, device='cuda')
            expected = dynamic_render(camera, model, pipe, background, indices, 1, 0,
                                      masks, tracks, transforms, separate_sh=True,
                                      est_mats_errors=errors)['render']
            original_xyz = model._xyz
            scene = SimpleNamespace(gaussians=model, model_path=directory)
            Scene.save_temporal(scene, 1, indices, tracks, transforms, masks, 1,
                                separate_sh=True, est_mats_errors=errors)
            self.assertIs(model._xyz, original_xyz)
            restored = GaussianModel(3)
            restored.load_ply(str(Path(directory, 'point_cloud/iteration_1/temporal_1.ply')))
            self.assertTrue(torch.isfinite(restored._opacity).all())
            actual = render(camera, restored, pipe, background, separate_sh=True)['render']
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
