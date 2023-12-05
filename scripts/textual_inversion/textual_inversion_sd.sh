#! /bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J dreamfusion
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=9:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=30G
##SBATCH --gpus=1

# module load gcc/7.5.0

# echo "===> Anaconda env loaded"
#source ~/.bashrc
#source activate magic123
# source venv_magic123/bin/activate 

# nvidia-smi
# nvcc --version

# hostname
# NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
# echo "number of gpus:" $NUM_GPU_AVAILABLE


MODEL_NAME="checkpoints/stable-diffusion-v1-5" # "path-to-pretrained-model" runwayml/stable-diffusion-v1-5
DATA_DIR="data/$2/image.jpg" # "path-to-dir-containing-your-image"
OUTPUT_DIR="outputs-textual-run/$2" # "path-to-desired-output-dir"
placeholder_token=$3 # _ironman_
init_token=$4 # ironman
# echo "1 param $1"
# echo "2 param $2"
# echo "3 param $3"
# echo "4 param $4"
echo "Placeholder Token $placeholder_token"

# run texturaal inversion
CUDA_VISIBLE_DEVICES=$1 accelerate launch textual-inversion/textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token=$placeholder_token \
  --initializer_token=$init_token \
  --resolution=512 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3000 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --use_augmentations \
  --only_save_embeds \
  --validation_prompt "A high-resolution DSLR image of ${placeholder_token}" \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision='fp16' \
  ${@:5}

# test textual inversion
# CUDA_VISIBLE_DEVICES=$1 python guidance/sd_utils.py --text "A high-resolution DSLR image of <token>" --learned_embeds_path $OUTPUT_DIR  --workspace $OUTPUT_DIR 