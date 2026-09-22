import sys
import torch
import os
import torchvision
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import shutil
import numpy as np

sys.path.append('./gaussian-splatting')
from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
from utils.camera_utils import cameraList_from_camInfos

from src.utils.localization_utils import hloc_localization, colmap_localization, readHlocCameras, save_hloc_results, hloc_results_from_colmap
from src.utils.visualization_utils import plot_rendering

def localize(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_localization : bool, known_intrinsics:bool, separate_sh: bool, ref_range=None, max_ref_candidates=None, min_inliers=20, query_camera_model=None, min_inlier_ratio=0.05, max_pose_references=None):
    with torch.no_grad():
        source_path = Path(dataset.source_path)
        scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
        output_dir = os.path.join("output", scene_name)
        hloc_result_dir = os.path.join(output_dir, "hloc")
        os.makedirs(hloc_result_dir, exist_ok=True)
        hloc_result_path =  os.path.join(hloc_result_dir, "hloc_results.json")

        if os.path.exists(os.path.join(dataset.model_path, "change")):
            shutil.rmtree(os.path.join(dataset.model_path, "change"))
        if not skip_localization:
            if str(source_path).endswith("hloc"):
                hloc_results = hloc_localization(dataset, known_intrinsics, ref_range=ref_range, max_ref_candidates=max_ref_candidates, min_inliers=min_inliers, query_camera_model=query_camera_model, min_inlier_ratio=min_inlier_ratio, max_pose_references=max_pose_references)
                # Checkpoint immediately: for large query sets this loop can take
                # many hours, and the render loop below is a separate failure
                # domain (GPU rendering) - don't lose the localization results
                # if rendering crashes partway through.
                save_hloc_results(hloc_results, hloc_result_path)
            else: # not used
                raise NotImplementedError("Use hloc for localization")
                # colmap_localization(dataset)
        else:
            dataset.single_timestep = False
            if str(source_path).endswith("hloc"):
                images_txt_path = source_path.parent / Path("images/changes.txt")
            else:
                images_txt_path = source_path / Path("images/changes.txt")
            with open(images_txt_path, 'r') as file:            
                images_path = file.read().strip().split()

        capture_path = os.path.join(output_dir, "change", "capture")
        render_path = os.path.join(output_dir, "change", "renders")
        os.makedirs(capture_path, exist_ok=True)
        os.makedirs(render_path, exist_ok=True)
        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if skip_localization:
            change_cameras = scene.getChangeCameras(images_path)  
        else:
            hloc_cameras = readHlocCameras(dataset, hloc_results, num_original_cameras=len(scene.train_cameras)+len(scene.test_cameras))
            change_cameras = cameraList_from_camInfos(hloc_cameras, 1.0, dataset, False, True)

        # Cap how many rendered/capture tensors plot_rendering ever sees - it
        # lays them out as len(renderings) matplotlib subplot columns, which is
        # only sane for a handful of images (the demo-scene scale this was
        # written for), not thousands of change cameras.
        MAX_PLOT_SAMPLES = 10
        renderings, captures = [], []
        for idx, view in enumerate(change_cameras):
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp, separate_sh=separate_sh)["render"]
            capture = view.original_image[0:3, :, :]
            if dataset.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                capture = view.original_image[0:3, :, :]

            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(capture, os.path.join(capture_path, '{0:05d}'.format(idx) + ".png"))

            # Move off GPU immediately - keeping every rendering/capture tensor
            # resident on GPU for the whole loop (as before) accumulates ~16MB
            # per change camera and exhausted VRAM partway through a ~4400-camera
            # run, crashing with a CUDA illegal memory access near the end.
            if idx < MAX_PLOT_SAMPLES:
                renderings.append(rendering.detach().cpu())
                captures.append(capture.detach().cpu())
            del rendering, capture

        renderings = torch.stack(renderings, dim=0)
        captures = torch.stack(captures, dim=0)

        plot_rendering(renderings, captures, output_dir, refined_renderings=None, iteration=0)

    if skip_localization:
        hloc_results = hloc_results_from_colmap(change_cameras)

    if str(source_path).endswith("hloc"):
        shutil.copy(source_path.parent / "images" / "changes.txt", os.path.join(hloc_result_dir, "changes.txt"))
    else:
        shutil.copy(source_path / "images" / "changes.txt", os.path.join(hloc_result_dir, "changes.txt"))
    
    save_hloc_results(hloc_results, hloc_result_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--min_inliers', type=int, default=20, help='Reject weak update poses before downstream processing.')
    parser.add_argument('--min_inlier_ratio', type=float, default=0.05)
    parser.add_argument('--max_pose_references', type=int, default=None,
                        help='Keep this many reference views ranked by confident matches for pose estimation.')
    parser.add_argument('--query_camera_model', choices=['SIMPLE_PINHOLE'], default=None,
                        help='Use for already rectified images; otherwise infer the camera model from metadata.')
    parser.add_argument("--skip_localization", action="store_true")
    # NOTE: plain `type=bool` is broken for a flag like this - argparse calls
    # bool("False") which is truthy, so "--known_intrinsics False" could never
    # actually disable it. BooleanOptionalAction gives real --known_intrinsics /
    # --no-known_intrinsics flags instead.
    parser.add_argument("--known_intrinsics", default=True, action=BooleanOptionalAction,
                         help="Assume query images share the reference reconstruction's camera "
                              "intrinsics (resolution/focal length). Pass --no-known_intrinsics "
                              "when queries come from different camera hardware or resolution "
                              "than the reference images, so intrinsics are inferred per image "
                              "instead.")
    parser.add_argument("--ref_range_start", default=None, type=float,
                         help="Only exhaustively match query images against reference images with a "
                              "timestamp >= this value (scanner filenames of the form "
                              "'<timestamp>_<idx>.jpg'). Omit for today's full-exhaustive behavior.")
    parser.add_argument("--ref_range_end", default=None, type=float,
                         help="Upper bound paired with --ref_range_start.")
    parser.add_argument("--max_ref_candidates", default=None, type=int,
                         help="Narrow each query to its top-K NetVLAD-retrieved reference images "
                              "before SuperGlue matching, instead of exhaustively matching against "
                              "every reference in range. Omit for today's full-exhaustive behavior; "
                              "set this for large reference sets where exhaustive matching is too slow.")

    args = get_combined_args(parser)
    print("HLOC Localization for " + args.model_path)

    ref_range_start = getattr(args, "ref_range_start", None)
    ref_range_end = getattr(args, "ref_range_end", None)
    if (ref_range_start is None) != (ref_range_end is None):
        raise ValueError("--ref_range_start and --ref_range_end must be given together")
    ref_range = (ref_range_start, ref_range_end) if ref_range_start is not None else None

    localize(model.extract(args), args.iteration, pipeline.extract(args), args.skip_localization, args.known_intrinsics, SPARSE_ADAM_AVAILABLE, ref_range=ref_range, max_ref_candidates=getattr(args, "max_ref_candidates", None), min_inliers=args.min_inliers, query_camera_model=args.query_camera_model, min_inlier_ratio=args.min_inlier_ratio, max_pose_references=args.max_pose_references)
