RUN_ID=$2
DYN_CFG=$3
SR_CFG=$4
DATA_DIR="data/$5"
PROMPT="$6, high-resolution, 8k, good quality"
CN_PROMPT="$7, high-resolution, 8k, good quality"
IMAGE_NAME='rgba.png'
FILENAME=$(basename $DATA_DIR)
dataset=$(basename $(dirname $DATA_DIR))


## -o equals to fp16 version
CUDA_VISIBLE_DEVICES=$1 python main.py -O \
--text "${PROMPT}" \
--hf_key checkpoints/text-to-video-ms-1.7b \
--workspace outputs-animate124/grid4d-magic123-${RUN_ID}/ms-cn-dynamic-image-${FILENAME}${DYN_CFG} \
--image ${DATA_DIR}/${IMAGE_NAME} \
--backbone grid4d \
--optim adam \
--iters 20000 \
--guidance SD zero123 CN \
--lambda_guidance 1.0 40 1.0 \
--guidance_scale 100 5 7.5 \
--grad_dyn \
--latent_iter_ratio 0 \
--normal_iter_ratio 0 \
--t_range 0.02 0.98 \
--zero123_t_range 0.2 0.6 \
--bg_radius -1 \
--albedo_iter_ratio 0 \
--h 128 --w 128 \
--time_size 16 \
--num_frames 16 \
--num_test_frames 64 \
--eval_interval 10 \
--test_interval 150 \
--lr_encoder_scale 10 \
--lr_sigma_scale 1 \
--lr_deform_scale 10 \
--grad_clip 1 \
--max_keep_ckpt 1 \
--static_ckpt outputs-animate124/grid4d-magic123-${RUN_ID}/static-${FILENAME}/checkpoints/static-${FILENAME}.pth \
--cn_base outputs-animate124/grid4d-magic123-${RUN_ID}/ms-dynamic-image-${FILENAME}${DYN_CFG}/checkpoints/ms-dynamic-image-${FILENAME}${DYN_CFG}.pth \
--dynamic_ft \
--dynamic_cam_rate 0.9 \
--no_view_text \
--grid_type tiledgrid \
--time_grid_size 64 \
--precision 256 \
--zero_rate 0.2 \
--lambda_time_tv 0.1 \
--cn_key checkpoints/controlnet-v11/control_v11f1e_sd15_tile \
--cn_learned_embeds_path  ${DATA_DIR}/learned_embeds.bin \
--cn_text "${CN_PROMPT}" \
--cn_frames 2 \
--cn_size 1024 \
--textureless_rate 0 \
--new_sds ${@:8}
