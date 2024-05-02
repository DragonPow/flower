# Federated Self-supervised Learning for Video Understanding

For specific downstream wit dataset [MOB](https://paperswithcode.com/dataset/mob), please run the following step to run this repo

### Setup environment

Please make sure you have installed CUDA 11.7 on your machine 
(see [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-11-7-0-download-archive)).
To construct the Python environment follow these steps (assuming you arelady have `pyenv` and `Poetry` in your system):

```bash
# Set directory to use python 3.10 (install with `pyenv install <version>` if you don't have it)
pyenv local 3.10.12
poetry env use 3.10.12

# Install the base Poetry environment
poetry install

# Activate the environment
poetry shell
```

### Setup dataset

1. Cloning repo `CtP`

```bash
# Clone CtP repo
git clone https://github.com/yan-gao-GY/CtP.git fedvssl/CtP

# Additional packages to decompress the dataset
sudo apt install unrar unzip
```

2. Preparing dataset

Let's first download MOB dataset and related annotation files:
```bash
cd fedvssl
mkdir -p data/mob/

# Downloading and put into data/mob/MOB_raw
```

3. Preprocessing dataset
```bash
python CtP/scripts/process_mob.py --raw_dir data/mob/MOB_raw/ --ann_dir data/mob/mobTrainTestlist/ --out_dir data/mob/

# Covert to .json files
python dataset_convert_to_json.py
```

4. Perform data partitioning for FL

```bash
python data_partitioning_ucf.py --json_path data/mob/annotations --output_path data/mob/annotations/client_distribution/ --num_clients 3

cd ..
```

### Federated SSL pre-training

Using model:
- [alpha0.9_beta0_round-540-weights.array.npz](https://drive.google.com/file/d/1W1oCnLXX0UJhQ4MlmRw-r7z5DTCeO75b/view?usp=sharing)
- [alpha0.9_beta1_round-540-weights.array.npz](https://drive.google.com/file/d/1BK-bbyunxTWNqs-QyOYiohaNv-t3-hYe/view?usp=sharing)

```bash
python -m fedvssl.finetune_preprocess --pretrained_model_path=<CHECKPOINT>.npz
# check for checkpoints in above items
```

### Fine tuning

```bash
bash fedvssl/CtP/tools/dist_train.sh fedvssl/conf/mmcv_conf/finetuning/r3d_18_ucf101/finetune_mob.py 1 --work_dir=./finetune_results --data_dir=fedvssl/data
```

After that, we perform the test process:

```bash
bash fedvssl/CtP/tools/dist_test.sh fedvssl/conf/mmcv_conf/finetuning/r3d_18_ucf101/test_mob.py 1 --work_dir=./finetune_results --data_dir=fedvssl/data --progress
```