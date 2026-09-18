"""Assemble an externally-trained 3DGS point cloud into a checkpoint directory the LTGS
pipeline scripts can load via -m, skipping train.py entirely."""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.append('./submodules/Hierarchical-Localization')


def infer_reference_prefix(sparse_dir):
    """If every registered image name in a COLMAP reconstruction follows a single
    <prefix><digits><ext> pattern (e.g. 'frame_00000.jpg'), return that shared prefix.
    Returns None if the names aren't consistent enough to infer one safely."""
    import pycolmap

    model = pycolmap.Reconstruction(str(sparse_dir))
    names = [model.images[i].name for i in model.reg_image_ids()]
    if not names:
        return None

    prefixes = set()
    for name in names:
        match = re.match(r'^(\D*)\d+$', Path(name).stem)
        if not match:
            return None
        prefixes.add(match.group(1))

    return prefixes.pop() if len(prefixes) == 1 else None


def write_checkpoint(pretrained_ply, checkpoint_dir, iteration, hloc_source_path, reference_prefix, dry_run):
    """Copy pretrained_ply into <checkpoint_dir>/point_cloud/iteration_<iteration>/point_cloud.ply
    and write a matching cfg_args, so downstream pipeline scripts can load this checkpoint via -m."""
    if dry_run:
        print(f"[dry-run] would write checkpoint under {checkpoint_dir} "
              f"(iteration {iteration}, reference_prefix={reference_prefix!r})")
        return

    checkpoint_dir = Path(checkpoint_dir)
    point_cloud_dir = checkpoint_dir / "point_cloud" / f"iteration_{iteration}"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pretrained_ply, point_cloud_dir / "point_cloud.ply")

    cfg = argparse.Namespace(
        sh_degree=3,
        source_path=str(Path(hloc_source_path).resolve()),
        model_path=str(checkpoint_dir.resolve()),
        images="images",
        depths="",
        resolution=-1,
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=True,
        single_timestep=False,
        reference_prefix=reference_prefix,
    )
    (checkpoint_dir / "cfg_args").write_text(str(cfg))
    print(f"Wrote checkpoint under {checkpoint_dir} (iteration {iteration})")
