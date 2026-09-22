#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import sys

sys.path.append('./gaussian-splatting')
from utils.loss_utils import ssim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    image_suffixes = {'.png', '.jpg', '.jpeg'}
    render_names = {p.name for p in renders_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes}
    gt_names = {p.name for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes}
    if not render_names or not gt_names:
        raise ValueError(f'No evaluation pairs in {renders_dir} and {gt_dir}. '
                         'All update images may be training inputs; an empty mean is not a score.')
    if render_names != gt_names:
        raise ValueError(f'Unpaired images: missing renders={gt_names-render_names}, missing GT={render_names-gt_names}')
    for fname in sorted(render_names):
        with Image.open(renders_dir / fname) as render:
            renders.append(tf.to_tensor(render.convert('RGB')).unsqueeze(0))
        with Image.open(gt_dir / fname) as gt:
            gts.append(tf.to_tensor(gt.convert('RGB')).unsqueeze(0))
        image_names.append(fname)
    return renders, gts, image_names

@torch.no_grad()
def evaluate(output_dir, split='test'):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = os.path.basename(output_dir)
    print("Scene:", scene_dir)
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(output_dir) / "update"

    gt_dir = test_dir / ("gt_all" if split == 'test' else "gt_train")
    renders_dir = test_dir / ("render_all" if split == 'test' else "render_train")

    time_indices = sorted([int(d.name) for d in gt_dir.iterdir() if d.is_dir()])

    psnrs, ssims, lpipss = [], [], []
    image_names_all = []
    criterion = None
    for time_idx in time_indices:
        if time_idx == 0: 
            continue

        renders, gts, image_names = readImages(renders_dir / str(time_idx), gt_dir / str(time_idx))
        if criterion is None:
            from lpipsPyTorch import LPIPS
            criterion = LPIPS(net_type='vgg').cuda().eval()

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            rendering, gt = renders[idx].cuda(), gts[idx].cuda()
            if rendering.shape != gt.shape:
                raise ValueError(f'Image dimensions differ: {image_names[idx]}')
            ssims.append(ssim(rendering, gt).item())
            psnrs.append(psnr(rendering, gt).item())
            lpipss.append(criterion(rendering, gt).item())
            del rendering, gt

        image_names_all.extend([os.path.join(str(time_idx), name) for name in image_names])

    if not image_names_all:
        raise ValueError('No update views were evaluated. Provide held-out images or use --split train with training renders.')
    if not all(torch.isfinite(torch.tensor(values)).all() for values in (ssims, psnrs, lpipss)):
        raise ValueError('Non-finite image metrics; refusing to write invalid JSON.')
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    per_view_dict[scene_dir].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names_all)},
                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names_all)},
                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names_all)}})

    full_dict[scene_dir].update({"SSIM": torch.tensor(ssims).mean().item(),
                                "PSNR": torch.tensor(psnrs).mean().item(),
                                "LPIPS": torch.tensor(lpipss).mean().item()})

    full_dict[scene_dir].update(num_views=len(image_names_all), evaluation_split=split)
    suffix = '' if split == 'test' else '_train'
    with open(output_dir + f"/results{suffix}.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True, allow_nan=False)
    with open(output_dir + f"/per_view{suffix}.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True, allow_nan=False)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--output_dir', '-m', required=True, type=str, default='')
    parser.add_argument('--split', choices=['test', 'train'], default='test', help='Train scores are diagnostics, not held-out evaluation.')
    args = parser.parse_args()
    evaluate(args.output_dir, args.split)
