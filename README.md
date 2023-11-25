# Animate124

This repository is the official implementation of **Animate124**.

**[Animate124: Animating One Image to 4D Dynamic Scene](https://arxiv.org/abs/2305.08850)**
<br/>
[Yuyang Zhao](https://yuyangzhao.com), [Zhiwen Yan](https://jokeryan.github.io/about/), [Enze Xie](https://xieenze.github.io/), [Lanqing Hong](https://scholar.google.com.sg/citations?user=2p7x6OUAAAAJ&hl=en), [Zhenguo Li](https://scholar.google.com.sg/citations?user=XboZC1AAAAAJ&hl=en), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/)
<br/>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://animate124.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2305.08850-b31b1b.svg)](https://arxiv.org/abs/2305.08850)

<p align="center">
<iframe width="640" height="360" src="https://www.youtube.com/embed/L_1HCBhz9MM?si=f4pn2uwajrc7BmrM"
        frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


## Abstract
> We introduce Animate124 (Animate-one-image-to-4D), the first work to animate a single in-the-wild image into 3D video through textual motion descriptions, an underexplored problem with significant applications. Our 4D generation leverages an advanced 4D grid dynamic Neural Radiance Field (NeRF) model, optimized in three distinct stages using multiple diffusion priors. Initially, a static model is optimized using the reference image, guided by 2D and 3D diffusion priors, which serves as the initialization for the dynamic NeRF. Subsequently, a video diffusion model is employed to learn the motion specific to the subject. However, the object in the 3D videos tends to drift away from the reference image over time. This drift is mainly due to the misalignment between the text prompt and the reference image in the video diffusion model. In the final stage, a personalized diffusion prior is therefore utilized to address the semantic drift. As the pioneering image-text-to-4D generation framework, our method demonstrates significant advancements over existing baselines, evidenced by comprehensive quantitative and qualitative assessments.


## TODO
- [ ] Release training code

## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{zhao2023animate124,
  author={Zhao, Yuyang and Yan, Zhiwen and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Lee, Gim Hee},
  title={Animate124: Animating One Image to 4D Dynamic Scene},
  journal={Arxiv},
  year={2023}
}
```

## Acknowledgements

This work is based on [Stable DreamFusion](https://github.com/ashawkey/stable-dreamfusion) and [Magic123](https://github.com/guochengqian/Magic123). If you use this code in your research, please also acknowledge their work.