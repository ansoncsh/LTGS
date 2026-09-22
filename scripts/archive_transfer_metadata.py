"""Archive AppleDouble sidecars produced by the source transfers in this investigation."""
from pathlib import Path
import shutil

repo = Path('/app/LTGS')
archive = Path('/opt/dlami/nvme/partial_update/debug/transfer_metadata')
files = ['metrics.py', 'localization.py', 'prepare_partial_update.py', 'long_term_update.py',
         'instance_matching.py', 'pcd_initialization.py', 'src/utils/hloc_reference.py',
         'src/utils/localization_utils.py', 'src/utils/make_depth_scale.py',
         'src/utils/matching_utils.py', 'src/utils/camera_intrinsics.py',
         'src/utils/registration_utils.py', 'tests/test_custom_dataset.py',
         'tests/test_temporal_export.py', 'gaussian-splatting/scene/__init__.py',
         'gaussian-splatting/scene/dataset_readers.py']
for filename in files:
    relative = Path(filename)
    sidecar = relative.with_name('._' + relative.name)
    source, target = repo / sidecar, archive / sidecar
    if source.is_file() and not source.is_symlink():
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)
        print(sidecar)
