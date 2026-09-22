"""Render a saved PLY at localized input views, explicitly as training diagnostics."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--ply', type=Path, required=True)
parser.add_argument('--poses', type=Path, required=True)
parser.add_argument('--images', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--min-inliers', type=int, default=0)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))
sys.path.insert(0, str(args.repo / 'gaussian-splatting'))
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import focal2fov

args.output.mkdir(parents=True, exist_ok=False)
gt_dir = args.output / 'update/gt_train/1'
render_dir = args.output / 'update/render_train/1'
gt_dir.mkdir(parents=True)
render_dir.mkdir(parents=True)
pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
gaussians = GaussianModel(3)
gaussians.load_ply(str(args.ply))
background = torch.zeros(3, device='cuda')
entries = json.loads(args.poses.read_text())
entries = [entry for entry in entries if entry['num_inliers'] is None or entry['num_inliers'] >= args.min_inliers]
if not entries:
    raise ValueError('No camera passed the requested inlier filter.')
manifest, rows = [], []
with torch.inference_mode():
    for idx, entry in enumerate(entries):
        params = entry['params']
        fx = params[0]
        fy = params[1] if entry['model'] == 'PINHOLE' else fx
        with Image.open(args.images / entry['name']) as original:
            resolution = (args.width, round(args.width * original.height / original.width))
            camera = Camera(resolution=resolution, colmap_id=idx,
                R=Rotation.from_quat(entry['rotation']).as_matrix().T, T=np.asarray(entry['translation']),
                FoVx=focal2fov(fx, entry['width']), FoVy=focal2fov(fy, entry['height']),
                depth_params=None, image=original.convert('RGB'), invdepthmap=None,
                image_name=entry['name'], uid=idx, data_device='cpu')
        result = render(camera, gaussians, pipe, background, separate_sh=True)['render'].clamp(0, 1)
        name = f'{idx:05d}.png'
        torchvision.utils.save_image(result, render_dir / name)
        torchvision.utils.save_image(camera.original_image[:3], gt_dir / name)
        manifest.append({'file': name, 'image_name': entry['name'], 'num_inliers': entry['num_inliers']})
        if len(entries) <= 7 or idx in (0, 3, 6, 9, 12, 15, 17):
            with Image.open(gt_dir / name) as gt, Image.open(render_dir / name) as rendered:
                thumb = (180, round(180 * gt.height / gt.width))
                row = Image.new('RGB', (360, thumb[1] + 25), 'white')
                row.paste(gt.resize(thumb), (0, 25))
                row.paste(rendered.resize(thumb), (180, 25))
                ImageDraw.Draw(row).text((4, 5), f'{entry["name"].split("/")[-1]}: GT / render', fill='black')
                rows.append(row)
        print(f'{idx+1}/{len(entries)} {entry["name"]}', flush=True)
        del result, camera
(args.output / 'view_manifest.json').write_text(json.dumps(manifest, indent=2))
montage = Image.new('RGB', (360 * len(rows), max(row.height for row in rows)), 'white')
for idx, row in enumerate(rows):
    montage.paste(row, (idx * 360, 0))
montage.save(args.output / 'comparison.jpg')
