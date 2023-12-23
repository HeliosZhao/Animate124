# Animate124 - threestudio

<img src="https://github.com/HeliosZhao/Animate124/assets/43061147/4c745058-d41f-47dc-aa98-25e06fe602d8" width="" height="200">
<img src="https://github.com/HeliosZhao/Animate124/assets/43061147/fcf3a7c1-62b4-458b-904e-2749411b7fa6" width="" height="200">

| [Project Page](https://animate124.github.io/) | [Paper](https://arxiv.org/pdf/2311.14603) | [Official Code](https://github.com/HeliosZhao/Animate124) |

This is Animate124 extension of [threestudio](https://github.com/threestudio-project/threestudio). The original implementation can be found [here](https://github.com/HeliosZhao/Animate124). The backbone (4D hash grid) and some hyperparameters of this implementation differ from those of the original one, so the results might be different. 

## Installation

### Install threestudio

**This part is the same as original threestudio. Skip it if you already have installed the environment.**

- You must have an NVIDIA graphics card with at least 24GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- (Optional, Recommended) Create a virtual environment:

```sh
python3 -m virtualenv venv
. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```

- Install `PyTorch >= 1.12`. We have tested on `torch2.0.1+cu117`, but other versions should also work fine.

```sh
# torch2.0.1+cu117
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install dependencies:

```sh
pip install -r requirements.txt
```


## Quickstart

Animate124 uses the default [textual inversion](https://huggingface.co/docs/diffusers/training/text_inversion) from diffuers. To run textual inversion: 

```sh
# Textual Inversion
gpu=0
CUSTOM_DIR="custom/threestudio-animate124"
MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATA_DIR="${CUSTOM_DIR}/load/panda-dance/image.jpg" # "path-to-dir-containing-your-image"
OUTPUT_DIR="outputs-textual-run/panda-dance" # "path-to-desired-output-dir"
placeholder_token="_panda_" 
init_token="_panda_" 
echo "Placeholder Token $placeholder_token"

CUDA_VISIBLE_DEVICES=$gpu accelerate launch ${CUSTOM_DIR}/textual-inversion/textual_inversion.py \
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
  --mixed_precision="fp16"
```
**Note:** Please move the final `learned_embeds.bin` under `custom/threestudio-animate124/load/$DATA_NAME/`

Animate124 is trained in 3 stages and there are three different config files for every stage. Training has to be resumed after finishing a stage. 

```sh
seed=0
gpu=0
exp_root_dir=outputs
DATA_DIR="panda-dance"
STATIC_PROMPT="a high resolution DSLR image of panda"
DYNAMIC_PROMPT="a panda is dancing"
CN_PROMPT="a <token> is dancing"

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

```

## Memory Usage
**Less VRAM**: The code requires 40GB VRAM in most cases. If you want to reduce the VRAM, you can try the following tricks:
- Reducing image size for semantic refinement (ControlNet Tile Guidance) in stage 3 (`system.guidance_cn.fixed_size`). If 32GB VRAM is available, solely setting `system.guidance_cn.fixed_size=896` should be enough.
- Reducing the number of ray samples with `system.renderer.num_samples_per_ray=128`.
- Reducing the rendering resolution for the video model with `data.single_view.width_vid=64` and `data.single_view.height_vid=64`.
- Setting `data.single_view.num_frames=8`, the number of frames can be reduced. (This might impair the performance a lot.)
- Reducing the hash grid capacity in system.geometry.pos_encoding_config, e.g., `system.geometry.pos_encoding_config.n_levels=8`. 
- Calculating gradient only for several video frames by setting `system.guidance.low_ram_vae`.


**More VRAM**: If you has 80GB VRAM, you can use the following parameters to achieve better performance: 
- Increasing the number of ray samples with `system.renderer.num_samples_per_ray=512`.
- Increasing the rendering resolution for the video model with `data.single_view.width_vid=256` and `data.single_view.height_vid=256`.
- Using ZeroScope video diffusion model as guidance. 
- Using float32 precision with `trainer.precision=32`.

## Credits
This code is built on the [threestudio-project](https://github.com/threestudio-project/threestudio) and [threestudio-4dfy](https://github.com/DSaurus/threestudio-4dfy). Thanks to the maintainers for their contribution to the community!

## Citing
If you find Animate124 helpful, please consider citing:

```
@article{zhao2023animate124,
  author    = {Zhao, Yuyang and Yan, Zhiwen and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Lee, Gim Hee},
  title     = {Animate124: Animating One Image to 4D Dynamic Scene},
  journal   = {arXiv preprint arXiv:2311.14603},
  year      = {2023},
}
```
