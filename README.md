# TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning

This repository contains the implementation of the AAAI2025 paper: TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning [[Paper]](https://arxiv.org/abs/2412.08176). 

## How to Run

We provide the running scripts in `scripts/textrefiner`, which allow you to reproduce the results on the paper.

Make sure you change the path in `DATA` and run the commands under the main directory `TextRefiner/`. In addition, you need to install an awesome toolbox dassl environment.

### Install

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n textrefiner python=3.8

# Activate the environment
conda activate textrefiner

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Clone TextRefiner code repository and install requirements
```bash
# Clone TextRefiner code base
git clone https://github.com/xjjxmu/TextRefiner.git


# Install requirements
cd TextRefiner/
pip install -r requirements.txt
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

### Datasets

Please follow the instructions at [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) in [CoOp](https://github.com/KaiyangZhou/CoOp) to prepare all datasets.

## Generalization From Base to New Classes

You will need both `scripts/textrefiner/base2new_train.sh` and `scripts/textrefiner/base2new_test.sh`. The former trains a model on base classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `TextRefiner/configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
# seed=1
bash scripts/textrefiner/base2new_train.sh imagenet 1
bash scripts/textrefiner/base2new_test.sh imagenet 1
```
For instance, after you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on ImageNet using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– TextRefiner/
|   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |–– seed1/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– TextRefiner/
|   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |–– seed1/
```


## Domain Generalization

The relevant scripts are `scripts/textrefiner/xd_train.sh` and `scripts/textrefiner/xd_test.sh` where the `DATASET` variable is set to the default, namely `imagenet`. To train the model, run
```bash
# seed=1
bash scripts/textrefiner/xd_train.sh 1
```

Then, you evaluate the model on other datasets, e.g.,

```bash
bash scripts/textrefiner/xd_test.sh imagenetv2 1
bash scripts/textrefiner/xd_test.sh imagenet_sketch 1
bash scripts/textrefiner/xd_test.sh imagenet_a 1
bash scripts/textrefiner/xd_test.sh imagenet_r 1
```

## Acknowledge

Our code and readme are based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [PromptKD](https://github.com/zhengli97/PromptKD) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well. 

# Citation

```
@misc{xie2024textrefinerinternalvisualfeature,
      title={TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning}, 
      author={Jingjing Xie and Yuxin Zhang and Jun Peng and Zhaohong Huang and Liujuan Cao},
      year={2024},
      eprint={2412.08176},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.08176}, 
}
```



### Concact

If you have any questions, you can submit an [issue](https://github.com/zhengli97/PromptKD/issues) on GitHub, or contact me by email (jingjingxie[at]stu.xmu.edu.cn).