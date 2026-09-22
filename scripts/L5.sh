DATA_DIR=/opt/dlami/nvme/partial_update
SCENE=L5
GPU=0

# prepare_partial_update.py output: DATA_DIR/ltgs_output/{images,hloc}
IMAGE_PATH=${DATA_DIR}/ltgs_output
# Checkpoint assembled from the externally-trained gaussians: cfg_args (source_path points
# at ${IMAGE_PATH}/hloc) + point_cloud/iteration_1/point_cloud.ply
OUTPUT=${DATA_DIR}/${SCENE}

# L5_art filenames encode a unix-timestamp prefix ("<ts>_<idx>.jpg"). Restrict which big-scene
# reference images HLoc exhaustively matches query images against (see localization.py
# --ref_range_start/--ref_range_end) to the window of the L5 walkthrough that actually passes
# by the L5_art capture area (previously covered by a black cloth during the L5 scan; starts
# becoming partially visible at 1785384085.227129, clearly framed through 1785384344.866413).
REF_RANGE_START=1785384085.227129
REF_RANGE_END=1785384344.866413
REF_RANGE_ARGS=()
if [[ -n "${REF_RANGE_START}" && -n "${REF_RANGE_END}" ]]; then
  REF_RANGE_ARGS=(--ref_range_start "${REF_RANGE_START}" --ref_range_end "${REF_RANGE_END}")
fi

# Even after ref_range narrows candidates to ~4,947 reference images, exhaustively
# SuperGlue-matching each of the 4,401 query images against all of them is far too
# slow (~10 days). Narrow further with NetVLAD retrieval to the top-K candidates
# per query before matching.
MAX_REF_CANDIDATES=30

# HLOC localization: localize the L5_art (images/changes.txt) query images against the big
# scene's SfM model. NOTE: unlike scripts/diningroom.sh this does NOT pass
# --skip_localization, since (unlike the demo scenes) localization has not been precomputed
# for L5 yet. --iteration 1 selects the single checkpoint under point_cloud/iteration_1.
CUDA_VISIBLE_DEVICES=${GPU} python localization.py -s ${IMAGE_PATH}/hloc -m ${OUTPUT} --iteration 1 "${REF_RANGE_ARGS[@]}" --max_ref_candidates ${MAX_REF_CANDIDATES}

# Change detection
CUDA_VISIBLE_DEVICES=${GPU} python change_detection.py -m ${OUTPUT} --min_size 1500 --kernel_size 5 --cosine_thr 0.93

# Instance matching
CUDA_VISIBLE_DEVICES=${GPU} python instance_matching.py -m ${OUTPUT} --fix_separated --filter_consistent --single_similarity_thres 0.92 --multi_similarity_thres 0.9 --optim_level refine+depth
CUDA_VISIBLE_DEVICES=${GPU} python pcd_initialization.py -m ${OUTPUT} --slackness 0

# Updating
CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m ${OUTPUT} --overlap_thres 0.1 --refine_iterations 2000 --conf_thres 2.5 --obj_pose_lr 0.001
python metrics.py -m ${OUTPUT}
