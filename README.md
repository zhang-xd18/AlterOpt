## Overview
This is a PyTorch implementation for the paper about the generalization challenge for DL-based CSI feedback, which has been submitted for IEEE for possible publication.

## Requirements

The following requirements need to be installed.
- Python == 3.9
- [PyTorch == 1.10.0](https://pytorch.org/get-started/previous-versions/#v1100)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from the [clustered delay line (CDL)](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173) model and settings can be found in our paper. 
% On the other hand, we provide a preprocessed dataset, which we adopt in the paper for testing the performance on avoiding catastrophic forgetting. You can download it from [Google Drive](https://drive.google.com/drive/folders/1yLzVBFR5rv3C_ym0PpDAnLyPOW5hiiCt?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1XDewsqmvFBAHNCoYtJVwwg) with the password: swgm.

#### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── AlterOpt  # The cloned current repository
│   ├── dataset
|   ├── pretrained # The checkpoints folder
|   |   ├── best_after_C.pth
|   |   ├── ...
│   ├── models
│   ├── utils
│   ├── main.py
|   ├── run.sh  # The bash script
├── 3GPP  # CDL dataset generated following section A
│   ├── DATA_HtestA.mat
│   ├── ...
...
```

A example of 'run.sh' is provided in the repository.

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Please refer to it for more information.
