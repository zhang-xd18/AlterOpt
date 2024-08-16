## Overview
This is a PyTorch implementation for the paper "Practical Deployment for Deep Learning-based CSI Feedback Systems: Generalization Challenges and Enabling Techniques", which has been submitted to IEEE for possible publication.

## Requirements

The following requirements need to be installed.
- Python == 3.9
- [PyTorch == 1.10.0](https://pytorch.org/get-started/previous-versions/#v1100)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from the [clustered delay line (CDL)](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173) model and settings can be found in our paper.  

#### B. Checkpoint and results downloading
The pretrained checkpoint used for online updating is provided in the "codes/pretrained" diectory along with the repository.

#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── code  # The cloned current repository
│   ├── dataset
|   ├── pretrained # The checkpoints folder
|   |   ├── best_after_C.pth
|   |   ├── ...
│   ├── models
│   ├── utils
│   ├── main.py
|   ├── run.sh  # The bash script
├── data  # CDL dataset generated following section A
│   ├── DATA_HtestA.mat
│   ├── ...
...
```
#### D. Key results reproduction

We test the performance of the alternating optimization method on both the CDL channel datasets. The results are presented as follows. 
##### Performance on the CDL dataset
All the results can be found in Figure 5 of our paper, which is also attached as CDL.png in the 'results' directory.

##### Reproduction
Before training the network, you need to arrange the dataset as abovementioned. After that, an example of `run.sh` is provided in the repository. The training procedure can be easily operated by `bash run.sh`.

``` bash
python /home/code/main.py \
    --epochs 200 \
    --gpu 0 \
    --root '/home/results/' \  # path to store the training results
    --name 'CRNet' \  # name of the network
    --data-dir '/home/data/' \  # path to the dataset 
    --batch-size 100 \
    --workers 0 \
    --cr 4 \
    --method 'Alter' \  # name of the online updating method
    --scenarios 'CDADA' \  # sequence of the scenario changing
    --pretrained '/home/code/pretrained/' \  # path to the offline pretrained checkpoints
    --store-num 100 \  # number of stored data for knowledge review n
    --period 2  # review period p
```
##### Performance on additional datasets
We also evaluate the performance of the alternating optimization framework on COST2100. The scenarios of SemiUrban_LOS, SemiUrban_NLOS, and IndoorHall are utilized. The results on COST2100 can be found in COST2100.PNG in the 'results' directory.

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Please refer to it for more information.
