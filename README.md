## Overview
This is a PyTorch implementation for the paper "Practical Deployment for Deep Learning-based CSI Feedback Systems: Generalization Challenges and Enabling Techniques", which has been submitted to IEEE for possible publication.

## Requirements

The following requirements need to be installed.
- Python >= 3.8
- [PyTorch == 1.13.0](https://pytorch.org/get-started/previous-versions/#v1130)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from the [clustered delay line (CDL)](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173) model and detailed settings can be found in our paper. On the other hand, we provide a preprocessed dataset, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1l0kJjXztHF03ojJFH4EFoo3YaXw81Jn_?usp=sharing) for your convenience.


#### B. Checkpoint and results downloading
The pretrained checkpoint used for online updating is provided in the `code/pretrain` directory along with the repository.

#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── code  # The cloned current repository
│   ├── dataset
|   ├── pretrain # The checkpoints folder
|   |   ├── best_after_C.pth
|   |   ├── ...
│   ├── models
│   ├── utils
│   ├── methods
│   ├── main.py
|   ├── run.sh  # The bash script
├── data  # CDL dataset generated following section A
│   ├── DATA_HtestA.mat
│   ├── ...
├── results  # Store the training results
...
```
#### D. Key results reproduction

We test the performance of the alternating optimization method on the CDL channel datasets. The results are presented as follows. 
##### Performance on the CDL dataset
All the results can be found in Figure 5 of our paper as follows, which is also attached as `CDL.png` in the `results` directory.
![image](https://github.com/zhang-xd18/AlterOpt/blob/main/results/CDL.png)

It can be seen that, with different parameter settings, AO can achieve more stable performance than OTL with scenario switching and higher performance than CL after each update. 

##### Reproduction
Before training the network, you need to arrange the dataset as abovementioned. After that, an example of `run.sh` is provided in the repository. The training procedure can be easily operated by `bash run.sh` after adjusting the file paths.

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
##### Ablation studies on additional datasets
###### Experiments on COST2100
We also evaluate the performance of the alternating optimization framework on COST2100. The scenarios of SemiUrban_LOS (SemiUrban scenario with LOS paths, working frequency of 300MHz), SemiUrban_NLOS (SemiUrban scenario with no LOS paths, working frequency of 300 MHz), and IndoorHall (IndoorHall scenario with LOS paths, working frequency of 5GHz) are utilized. The datasets can be generated according to the [open source library of COST2100](https://github.com/cost2100/cost2100). 

The training settings are similar to that on the CDL channel model. The model is first sufficiently offline trained with the scenario of SemiUrban_LOS, achieving an NMSE of -8.996 dB after 2,500 epochs of training with 50,000 samples. After deployment, only 200 epochs of online updating and 50 epochs of knowledge review are conducted. Finally, the results on COST2100 can be found in `COST2100.png` in the `results` directory.
![image](https://github.com/zhang-xd18/AlterOpt/blob/main/results/COST2100.png)

It can be seen that our proposed AO method can achieve more stable performances than OTL during scenario switching. Besides, with a smaller period p, the AO can achieve more stable performance over different scenarios.

###### Experiments on channel model switch
We also explore the performance of the alternating optimization framework on variations among different channel models. We utilize the COST2100 and CDL channel models. For the COST2100 channel model, we use the IndoorHall scenario with LOS paths working on 5GHz. For the CDL channel model, we consider the CDL-A scenario. 

The model is initially trained offline under the CDL-C scenario of CDL channel model, with -17.76dB of NMSE on CDL-C.
During online updating, only 2,500 samples are collected for 200 epochs of online updating. For knowledge review, only 50 epochs are conducted.
The results of channel model switch can be found in `Model_change.png` in the `results` directory.
![image](https://github.com/zhang-xd18/AlterOpt/blob/main/results/Model_change.png)

It can be seen that with knowledge review on previous channel models, our proposed AO framework can maintain more stable and effective performance compared to OTL, demonstrating a higher generalization ability.

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Please refer to it for more information.
