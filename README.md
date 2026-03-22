# Reframing Gaussian Splatting Densification with Complexity-Density Consistency of Primitive

Accepted by NeurIPS 2025

### [Webpage](https://cdc-gs.github.io/) | [Paper](https://openreview.net/pdf?id=VKJTyhAtoA)

This repository contains the official authors implementation associated with the paper "Reframing Gaussian Splatting Densification with Complexity-Density Consistency of Primitive".

## Environment Setup
To prepare the environment, 

1. Clone this repository. 
    ```
    git clone https://github.com/ZhemengDong/CDC-GS.git
    ```
2. Follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install dependencies. 
    ```
    conda env create --file environment.yml
    conda activate cdcgs
    ```
3. Install Submodules:
    ```
    CUDA_HOME=PATH/TO/CONDA/envs/3dgs-mcmc-env/pkgs/cuda-toolkit/ pip install submodules/diff-gaussian-rasterization submodules/diff-gaussian-rasterization-cdc submodules/simple-knn/
        ```

## Run CDC-GS
Train 3DGS
  ```
    python train.py -s "dataset/tandt/train" -m "output/train"
  ```
  
## Citation
*If you find this project helpful for your research, please consider citing the report and giving a ⭐.*

*Any questions are welcome for discussion.*
```
@inproceedings{
dong2025reframing,
title={Reframing Gaussian Splatting Densification with Complexity-Density Consistency of Primitives},
author={Zhemeng Dong and Junjun Jiang and Youyu Chen and Jiaxin Zhang and Kui Jiang and Xianming Liu},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=VKJTyhAtoA}
}
```
