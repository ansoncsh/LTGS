"""Assemble the LTGS pipeline's expected data layout (<scene>/images/changes.txt +
<scene>/hloc/{features.h5,matches.h5,pairs-netvlad.txt,sparse/0/*.bin}) from a big scene (the
existing large scan) and a small scene (a sparse-view update capture of one region of it).

Each side's source can be provided in either of two ways:
  * --big_scene_dir / --small_scene_dir: a raw scanner export in this project's existing
    capture layout (see find_perspective_dir/validate_perspective_dir below for the exact
    folder structure expected)
  * --big_scene_images/--big_scene_sparse (and --small_scene_images): an explicit images
    directory and, for the big scene, a COLMAP reconstruction (cameras.bin/images.bin/
    points3D.bin, found at any nesting depth) - each of these may be a directory or a .zip
    file, which is extracted before use.

Optionally, --pretrained_ply assembles an externally-trained 3DGS checkpoint (skipping
train.py) into <checkpoint_dir>/{cfg_args,point_cloud/iteration_<N>/point_cloud.ply}, which
downstream pipeline scripts (localization.py, change_detection.py, ...) load via -m. Getting
cfg_args' reference_prefix wrong makes Scene loading silently drop every big-scene camera
instead of erroring, so it is inferred from the reconstruction's registered image names
when not given explicitly. See src/utils/checkpoint_utils.py for that part.

Usage:
  python prepare_partial_update.py \\
      --big_scene_dir data/big_scene --small_scene_dir data/small_scene \\
      --output_dir data/ltgs_dataset/my_scene

  python prepare_partial_update.py \\
      --big_scene_images big_scene/images.zip --big_scene_sparse big_scene/colmap.zip \\
      --small_scene_images small_scene/images \\
      --pretrained_ply big_scene/point_cloud.ply --output_dir data/ltgs_dataset/my_scene
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from src.utils.checkpoint_utils import infer_reference_prefix, write_checkpoint

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Needed by build_hloc_export()'s pycolmap/hloc imports below. Appending the path is cheap
# (no heavy import triggered) so it's done unconditionally here rather than repeated inside
# every function that needs it.
sys.path.append('./submodules/Hierarchical-Localization')

# Set once from argv in main() and read by the assembly helpers below, instead of threading
# a dry_run bool through every one of their signatures.
DRY_RUN = False


def find_perspective_dir(scene_dir):
    matches = sorted(glob.glob(os.path.join(scene_dir, "**", "developer_data", "perspective"), recursive=True))
    if len(matches) == 0:
        raise FileNotFoundError(f"No 'developer_data/perspective' folder found under {scene_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple 'developer_data/perspective' folders found under {scene_dir}: {matches}")
    return Path(matches[0])


def validate_perspective_dir(perspective_dir):
    images_dir = perspective_dir / "images"
    sparse_dir = perspective_dir / "sparse"
    missing = [p for p in [images_dir, sparse_dir] if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"{perspective_dir} is missing expected subfolder(s): {missing}")
    for required in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse_dir / required).exists():
            raise FileNotFoundError(f"{sparse_dir} is missing {required}")
    return images_dir, sparse_dir


def extract_zip(zip_path, extract_root, label):
    target = extract_root / label
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    return target


def resolve_source_dir(path, extract_root, label):
    """Accept either a directory or a .zip file, extracting the latter under extract_root.
    Returns (resolved_dir, was_extracted_from_zip)."""
    path = Path(path)
    if path.is_dir():
        return path, False
    if path.suffix.lower() == ".zip":
        return extract_zip(path, extract_root, label), True
    raise ValueError(f"{path} is neither a directory nor a .zip file")


def find_colmap_sparse_dir(root):
    """Locate the folder containing a COLMAP reconstruction's cameras.bin/images.bin/
    points3D.bin under root, at any nesting depth."""
    candidates = sorted({p.parent for p in Path(root).rglob("images.bin")})
    candidates = [p for p in candidates if (p / "cameras.bin").exists() and (p / "points3D.bin").exists()]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No COLMAP reconstruction (cameras.bin/images.bin/points3D.bin) found under {root}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple COLMAP reconstructions found under {root}: {candidates}")
    return candidates[0]


def resolve_scene(raw_dir_arg, images_arg, sparse_arg, extract_root, label, needs_sparse):
    """Resolve one scene's images (and, for the big scene, sparse reconstruction) from either
    the images_arg/sparse_arg zip-or-dir mode, or the raw_dir_arg scanner-export mode.
    Returns (images_dir, sparse_dir_or_None, scene_name, any_zip_source)."""
    if images_arg:
        images_dir, any_zip = resolve_source_dir(images_arg, extract_root, f"{label.lower()}_images")
        sparse_dir = None
        if needs_sparse:
            sparse_root, sparse_from_zip = resolve_source_dir(sparse_arg, extract_root, f"{label.lower()}_sparse")
            any_zip = any_zip or sparse_from_zip
            sparse_dir = find_colmap_sparse_dir(sparse_root)
        scene_name = Path(images_arg).stem
    else:
        scene_dir = Path(raw_dir_arg)
        perspective = find_perspective_dir(scene_dir)
        print(f"{label} scene perspective dir: {perspective}")
        images_dir, found_sparse_dir = validate_perspective_dir(perspective)
        sparse_dir = found_sparse_dir if needs_sparse else None
        scene_name = scene_dir.name
        any_zip = False

    return images_dir, sparse_dir, scene_name, any_zip


def list_images(images_dir):
    files = []
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return files


def place_file(src, dst, link):
    if DRY_RUN:
        print(f"  [dry-run] {'symlink' if link else 'copy'}: {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def assemble_images(big_images_dir, small_images_dir, small_scene_name, out_images_dir, link):
    big_files = list_images(big_images_dir)
    print(f"Big scene: {len(big_files)} images under {big_images_dir}")
    for src in big_files:
        rel = src.relative_to(big_images_dir)
        place_file(src, out_images_dir / rel, link)

    small_files = list_images(small_images_dir)
    print(f"Small scene: {len(small_files)} images under {small_images_dir}")
    change_rel_paths = []
    for src in small_files:
        rel = src.relative_to(small_images_dir)
        out_rel = Path(small_scene_name) / rel
        place_file(src, out_images_dir / out_rel, link)
        change_rel_paths.append(out_rel.as_posix())

    return sorted(change_rel_paths)


def write_changes_txt(out_images_dir, change_rel_paths):
    changes_path = out_images_dir / "changes.txt"
    content = "\n".join(change_rel_paths) + "\n"
    if DRY_RUN:
        print(f"[dry-run] would write {changes_path} ({len(change_rel_paths)} entries)")
        return
    out_images_dir.mkdir(parents=True, exist_ok=True)
    changes_path.write_text(content)
    print(f"Wrote {changes_path} ({len(change_rel_paths)} entries)")


def assemble_sparse(big_sparse_dir, out_hloc_dir):
    out_sparse0 = out_hloc_dir / "sparse" / "0"
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        place_file(big_sparse_dir / name, out_sparse0 / name, link=False)

    # train.py (standard 3DGS) expects <source_path>/images/ directly, while
    # hloc_localization() expects <source_path>/../images/ (source_path == .../hloc). Since
    # dataset.source_path must be this hloc/ folder (its name is what gates the hloc code path),
    # give it its own images/ symlink pointing back at the shared images/ folder.
    out_images_link = out_hloc_dir / "images"
    if DRY_RUN:
        print(f"  [dry-run] symlink: {out_hloc_dir}/images -> ../images")
    else:
        out_hloc_dir.mkdir(parents=True, exist_ok=True)
        if not out_images_link.exists() and not out_images_link.is_symlink():
            out_images_link.symlink_to(Path("..") / "images")

    return out_sparse0


def build_hloc_export(out_images_dir, out_sparse0, out_hloc_dir, num_matched=20):
    if DRY_RUN:
        print(f"[dry-run] would build hloc export (features.h5, matches.h5, pairs-netvlad.txt) under {out_hloc_dir}")
        return

    import pycolmap
    from hloc import extract_features, match_features, pairs_from_retrieval

    model = pycolmap.Reconstruction(out_sparse0)
    references_registered = [model.images[i].name for i in model.reg_image_ids()]
    print(f"Building HLoc SfM export for {len(references_registered)} registered big-scene images")

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    out_hloc_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = extract_features.main(
        retrieval_conf, out_images_dir, image_list=references_registered, feature_path=out_hloc_dir / "global-features.h5"
    )
    pairs_path = out_hloc_dir / "pairs-netvlad.txt"
    pairs_from_retrieval.main(retrieval_path, pairs_path, num_matched=min(num_matched, len(references_registered)))

    feature_path = extract_features.main(
        feature_conf, out_images_dir, image_list=references_registered, feature_path=out_hloc_dir / "features.h5"
    )
    match_features.main(matcher_conf, pairs_path, features=feature_path, matches=out_hloc_dir / "matches.h5")
    # Original COLMAP tracks index its original features, not the newly extracted
    # SuperPoint features. Keep the original reconstruction for scene loading and
    # build a separate, fixed-pose reference model for HLoc's index-based lookup.
    from src.utils.hloc_reference import triangulate_reference
    triangulate_reference(out_hloc_dir, out_images_dir)
    print(f"HLoc SfM export written under {out_hloc_dir}")


def main():
    global DRY_RUN

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--big_scene_dir", default=None, help="Root dir of the big scene's raw scanner export")
    parser.add_argument("--small_scene_dir", default=None, help="Root dir of the small update capture's raw scanner export")
    parser.add_argument("--big_scene_images", default=None,
                         help="Directory or .zip of the big scene's images; alternative to --big_scene_dir")
    parser.add_argument("--big_scene_sparse", default=None,
                         help="Directory or .zip containing the big scene's COLMAP reconstruction "
                              "(cameras.bin/images.bin/points3D.bin, at any nesting depth); "
                              "required together with --big_scene_images")
    parser.add_argument("--small_scene_images", default=None,
                         help="Directory or .zip of the update capture's images; alternative to --small_scene_dir")
    parser.add_argument("--output_dir", default=None, help="Defaults to data/ltgs_dataset/<scene name>")
    parser.add_argument("--link", action="store_true",
                         help="Symlink images instead of copying them (ignored - forced to copy - for any "
                              "side sourced from a .zip, since the extracted files don't persist)")
    parser.add_argument("--dry_run", action="store_true", help="Log actions without touching disk")
    parser.add_argument("--num_matched", type=int, default=20, help="Retrieval pairs per image for the big scene's HLoc SfM export")
    parser.add_argument("--pretrained_ply", default=None,
                         help="Optional pretrained 3DGS point_cloud.ply to assemble into a checkpoint "
                              "directory (see --checkpoint_dir), skipping train.py")
    parser.add_argument("--checkpoint_dir", default=None,
                         help="Output dir for the assembled checkpoint (with --pretrained_ply); "
                              "defaults to output/<scene name>")
    parser.add_argument("--checkpoint_iteration", type=int, default=1,
                         help="Iteration number under the checkpoint's point_cloud/ directory")
    parser.add_argument("--reference_prefix", default=None,
                         help="Filename prefix identifying the big scene's images (e.g. 'frame_'), written "
                              "into the checkpoint's cfg_args; inferred automatically from the reconstruction "
                              "when every registered image follows a single <prefix><number> pattern")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    if bool(args.big_scene_images) != bool(args.big_scene_sparse):
        parser.error("--big_scene_images and --big_scene_sparse must be given together")
    if not args.big_scene_dir and not args.big_scene_images:
        parser.error("Provide either --big_scene_dir or both --big_scene_images and --big_scene_sparse")
    if not args.small_scene_dir and not args.small_scene_images:
        parser.error("Provide either --small_scene_dir or --small_scene_images")

    with tempfile.TemporaryDirectory() as extract_root:
        extract_root = Path(extract_root)

        big_images_dir, big_sparse_dir, scene_name, big_any_zip = resolve_scene(
            args.big_scene_dir, args.big_scene_images, args.big_scene_sparse, extract_root, "Big", needs_sparse=True
        )
        small_images_dir, _, small_scene_name, small_any_zip = resolve_scene(
            args.small_scene_dir, args.small_scene_images, None, extract_root, "Small", needs_sparse=False
        )
        any_zip_source = big_any_zip or small_any_zip

        output_dir = Path(args.output_dir) if args.output_dir else Path("data/ltgs_dataset") / scene_name
        out_images_dir = output_dir / "images"
        out_hloc_dir = output_dir / "hloc"

        link = args.link and not any_zip_source
        if args.link and not link:
            print("Ignoring --link: at least one input came from a .zip, so images will be copied instead of symlinked")

        change_rel_paths = assemble_images(big_images_dir, small_images_dir, small_scene_name, out_images_dir, link)
        write_changes_txt(out_images_dir, change_rel_paths)
        out_sparse0 = assemble_sparse(big_sparse_dir, out_hloc_dir)

        reference_prefix = args.reference_prefix
        if args.pretrained_ply and reference_prefix is None and not DRY_RUN:
            reference_prefix = infer_reference_prefix(big_sparse_dir)
            if reference_prefix is None:
                parser.error("Could not infer --reference_prefix from the big scene's registered image "
                              "names (they don't all follow a single <prefix><number> pattern); pass it explicitly")
            print(f"Inferred --reference_prefix={reference_prefix!r}")

    build_hloc_export(out_images_dir, out_sparse0, out_hloc_dir, args.num_matched)
    print(f"\nDone. Pipeline data ready under: {output_dir}")

    if args.pretrained_ply:
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("output") / scene_name
        write_checkpoint(args.pretrained_ply, checkpoint_dir, args.checkpoint_iteration,
                          out_hloc_dir, reference_prefix, DRY_RUN)


if __name__ == "__main__":
    main()
