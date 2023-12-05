# Animate124

This repository is the official implementation of **Animate124**.

**[Animate124: Animating One Image to 4D Dynamic Scene](https://arxiv.org/abs/2311.14603)**
<br/>
[Yuyang Zhao](https://yuyangzhao.com), [Zhiwen Yan](https://jokeryan.github.io/about/), [Enze Xie](https://xieenze.github.io/), [Lanqing Hong](https://scholar.google.com.sg/citations?user=2p7x6OUAAAAJ&hl=en), [Zhenguo Li](https://scholar.google.com.sg/citations?user=XboZC1AAAAAJ&hl=en), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/)
<br/>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://animate124.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2311.14603-b31b1b.svg)](https://arxiv.org/abs/2311.14603)


https://github.com/HeliosZhao/Animate124/assets/43061147/282e5855-8c40-4375-ab29-f23a18952c1b



## Abstract
> We introduce Animate124 (Animate-one-image-to-4D), the first work to animate a single in-the-wild image into 3D video through textual motion descriptions, an underexplored problem with significant applications. Our 4D generation leverages an advanced 4D grid dynamic Neural Radiance Field (NeRF) model, optimized in three distinct stages using multiple diffusion priors. Initially, a static model is optimized using the reference image, guided by 2D and 3D diffusion priors, which serves as the initialization for the dynamic NeRF. Subsequently, a video diffusion model is employed to learn the motion specific to the subject. However, the object in the 3D videos tends to drift away from the reference image over time. This drift is mainly due to the misalignment between the text prompt and the reference image in the video diffusion model. In the final stage, a personalized diffusion prior is therefore utilized to address the semantic drift. As the pioneering image-text-to-4D generation framework, our method demonstrates significant advancements over existing baselines, evidenced by comprehensive quantitative and qualitative assessments.

## News
- [05/12/2023] Code Released!
## TODO
- [ ] Implementation with [threestudio](https://github.com/threestudio-project/threestudio)
- [x] Release training code


## Install Environment 
We use torch 2.0.1 and xformers 0.0.21 in this project.

```bash
bash install.sh
```


### Download pre-trained models

Download the following pre-trained models from huggingface and put them into `checkpoints`:
- [StableDiffusion-v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
- [Zero-1-to-3-XL](https://huggingface.co/Yuyang-z/zero123-xl)
- [ModelScope](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)
- [ControlNet-v1.1-Tile](https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile)

Download [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation
  ```bash
  mkdir -p pretrained/midas
  cd pretrained/midas
  wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
  cd ../../
  ```

## Run

The whole training process includes 4 steps: textual inversion, static stage, dynamic coarse stage and dynamic fine stage. The first three steps require 40GB VRAM while the final step requires 80GB VRAM.

### Preprocess
We have included all preprocessed files in `./data` directory. Preprocessing is only necessary if you want to test on your own examples. Takes seconds.   
```
python preprocess_image.py --path /path/to/image 
```


#### Step 1: textual inversion 
Animate124 uses the default [textual inversion](https://huggingface.co/docs/diffusers/training/text_inversion) from diffuers. To run textual inversion: 

```
bash scripts/textual_inversion/textual_inversion_sd.sh $GPU $DATA_NAME $TOKEN_NAME $TOKEN_INIT
```
`$TOKEN_NAME` is a the special token, usually name that by _examplename_

`$TOKEN_INIT` is a single token to describe the image using natural language

For example:
```bash
bash scripts/textual_inversion/textual_inversion_sd.sh $GPU panda-dance _panda_ panda
```
**Note:** Please move the final `learned_embeds.bin` under `data/$DATA_NAME/`

#### Step 2: Static Stage

```bash
bash scripts/animate124/run-static.sh $GPU run $DATA_NAME $PROMPT
```

For example:
```bash
bash scripts/animate124/run-static.sh $GPU run panda-dance "a high-resolution DSLR image of a panda"
```

#### Step 3: Dynamic Coarse Stage

```bash
bash scripts/animate124/run-dynamic.sh $GPU run run $DATA_NAME $PROMPT
```

For example:
```bash
bash scripts/animate124/run-dynamic.sh $GPU run run panda-dance "a panda is dancing"
```

#### Step 4: Dynamic Fine Stage (Semantic Refinement)

```bash
bash scripts/animate124/run-cn.sh $GPU run run run $DATA_NAME $PROMPT $CN_PROMPT
```

For example:
```bash
bash scripts/animate124/run-cn.sh $GPU run run run panda-dance "a panda is dancing" "a <token> is dancing"
```

The overall training scripts for examples in `./data` are in `scripts/run_data`

## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{zhao2023animate124,
  author={Zhao, Yuyang and Yan, Zhiwen and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Lee, Gim Hee},
  title={Animate124: Animating One Image to 4D Dynamic Scene},
  journal={arXiv preprint arXiv:2311.14603},
  year={2023}
}
```

## Acknowledgements

This work is based on [Stable DreamFusion](https://github.com/ashawkey/stable-dreamfusion) and [Magic123](https://github.com/guochengqian/Magic123). If you use this code in your research, please also acknowledge their work.
