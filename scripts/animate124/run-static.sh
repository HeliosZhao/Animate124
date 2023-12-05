RUN_ID=$2
DATA_DIR="data/${3}"
PROMPT="${4}, 8k, good quality"
IMAGE_NAME='rgba.png'
FILENAME=$(basename $DATA_DIR)
dataset=$(basename $(dirname $DATA_DIR))

## -o equals to fp16 version
CUDA_VISIBLE_DEVICES=$1 python main.py -O \
--text "${PROMPT}" \
--sd_version 1.5 \
--image ${DATA_DIR}/${IMAGE_NAME} \
--workspace outputs-animate124/grid4d-magic123-${RUN_ID}/static-${FILENAME} \
--backbone grid4d \
--optim adam \
--iters 5000 \
--guidance SD zero123 \
--lambda_guidance 1.0 40 \
--guidance_scale 100 5 \
--latent_iter_ratio 0 \
--normal_iter_ratio 0.2 \
--t_range 0.2 0.6 \
--bg_radius -1 \
--grid_type tiledgrid ${@:5}

## t range both set to 0.2 ~ 0.6 is better