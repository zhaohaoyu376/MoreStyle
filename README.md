# MoreStyle: Relax Low-frequency Constraint of Fourier-based Image Reconstruction in Generalizable Medical Image Segmentation **[MICCAI 2024]**

## Introduction
The task of single-source domain generalization (SDG) in medical image segmentation is crucial due to frequent domain shifts in clinical image datasets. To address the challenge of poor generalization across different domains, we introduce a Plug-and-Play module for data augmentation called MoreStyle. MoreStylediversifies image styles by relaxing low-frequency constraints in Fourier space, guiding the image reconstruction network. With the help of adversarial learning, MoreStylefurther expands the style range and pinpoints the most intricate style combinations within latent features. To handle significant style variations, we introduce an uncertainty-weighted loss. This loss emphasizes hard-to-classify pixels resulting only from style shifts while mitigating true hard-to-classify pixels in both MoreStyle-generated and original images. Extensive experiments on two widely used benchmarks demonstrate that the proposed MoreStyle effectively helps to achieve good domain generalization ability, and has the potential to further boost the performance of some state-of-the-art SDG methods.

## Prerequisites
- Linux
- Python 3.7
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/zhaohaoyu376/MoreStyle
cd MoreStyle
```

- Install [PyTorch](http://pytorch.org) and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).

## Datasets
The RIGA+ dataset can be downloaded from: [RIGA+](https://zenodo.org/records/6325549).

The prostate dataset can be downloaded from: [prostate](http://medicaldecathlon.com/).

### MoreStyle train/test
- train the model:
```bash
# for RIGA+ dataset
python train_segmentation_fourier.py
# for prostate dataset
python train_PROSTATE_fourier.py
```


## Citation
If you use this code for your research, please cite our papers.
```
@article{zhao2024morestyle,
  title={MoreStyle: Relax Low-frequency Constraint of Fourier-based Image Reconstruction in Generalizable Medical Image Segmentation},
  author={Zhao, Haoyu and Dong, Wenhui and Yu, Rui and Zhao, Zhou and Bo, Du and Xu, Yongchao},
  journal={arXiv preprint arXiv:2403.11689},
  year={2024}
}
```

## Acknowledgments
Our code is inspired by [CCSDG](https://github.com/ShishuaiHu/CCSDG).
