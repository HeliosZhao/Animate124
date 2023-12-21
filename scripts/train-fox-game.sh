seed=$2
gpu=$1
exp_root_dir=outputs
DATA_DIR="fox-game"
STATIC_PROMPT="a high-resolution DSLR image of a fox maneuvering a game controller with its paws"
DYNAMIC_PROMPT="a fox maneuvering a game controller with its paws"
CN_PROMPT="a <token> maneuvering a game controller with its paws"

# --------- Stage 1 (Static Stage) --------- #
python launch.py --config custom/threestudio-animate124/configs/animate124-stage1.yaml --train --gpu $gpu \
data.image.image_path=custom/threestudio-animate124/load/${DATA_DIR}/_rgba.png \
system.prompt_processor.prompt="${STATIC_PROMPT}"

# --------- Stage 2 (Dynamic Coarse Stage) --------- #
ckpt=outputs/animate124-stage1/${STATIC_PROMPT}@LAST/ckpts/last.ckpt
python launch.py --config custom/threestudio-animate124/configs/animate124-stage2-ms.yaml --train --gpu $gpu \
data.image.image_path=custom/threestudio-animate124/load/${DATA_DIR}/_rgba.png \
system.prompt_processor.prompt="${DYNAMIC_PROMPT}" \
system.weights="$ckpt"

# --------- Stage 2 (Semantic Refinement Stage) --------- #
ckpt=outputs/animate124-stage2/${DYNAMIC_PROMPT}@LAST/ckpts/last.ckpt
python launch.py --config custom/threestudio-animate124/configs/animate124-stage3-ms.yaml --train --gpu $gpu \
data.image.image_path=custom/threestudio-animate124/load/${DATA_DIR}/_rgba.png \
system.prompt_processor.prompt="${DYNAMIC_PROMPT}" \
system.prompt_processor_cn.prompt="${CN_PROMPT}" \
system.prompt_processor_cn.learned_embeds_path=custom/threestudio-animate124/load/${DATA_DIR}/learned_embeds.bin \
system.weights="$ckpt" 