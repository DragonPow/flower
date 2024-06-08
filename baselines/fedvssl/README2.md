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

# or 
python data_partitioning_ucf.py --input_file data/mob/annotations/train_split_1.json --output_path data/mob/annotations/client_distribution/ --num_clients 3

cd ..
```

Workdir = `/fedvssl/mmcv_conf/`

Base config file
```yaml
# base.yaml
pool_size: 2 # number of client
rounds: 2 # number of round for global server
...
strategy:
  min_fit_clients: 2 # number of client join set parameter when compute in server
  min_available_clients: 2 # min of client need to ready to start every round
...
client_resources:
  num_gpus: 1
  num_cpus: 2
```

Specific model config
```py
# pretraining/pretraining_runtime_mob.py
total_epochs = 10 # epoch for every client in 1 round
lr_config = {"policy": "step", "step": [100, 200]} # learning policy, with step is mean decrease lr every n epoch
```

### Federated SSL pre-training

1. Using exists model:
- [alpha0.9_beta0_round-540-weights.array.npz](https://drive.google.com/file/d/1W1oCnLXX0UJhQ4MlmRw-r7z5DTCeO75b/view?usp=sharing)
- [alpha0.9_beta1_round-540-weights.array.npz](https://drive.google.com/file/d/1BK-bbyunxTWNqs-QyOYiohaNv-t3-hYe/view?usp=sharing)

2. Or using custom pretraining model by above command:

```bash
python -m fedvssl.main
```

3. After that, preprocess file result
```bash
python -m fedvssl.finetune_preprocess --pretrained_model_path=<CHECKPOINT>.npz
# With checkpoint a the output file .npz of 1. or 2. above
```

### Fine tuning

1. Custom config

Workdir = `/fedvssl/mmcv_conf/`

```py
# finetuning/runtime_mob.py
total_epochs = 10 # epoch for every client in 1 round
lr_config = {"policy": "step", "step": [5, 10, 13], "gamma": 0.5} # learning policy, with step is mean decrease lr every n epoch, learning rate = gramma * learning rate
```

2. Running file tuning

```bash
# Linux
bash fedvssl/CtP/tools/dist_train.sh fedvssl/conf/mmcv_conf/finetuning/r3d_18_ucf101/finetune_mob.py 1 --work_dir=./finetune_results --data_dir=fedvssl/data

# Windows
./fedvssl/CtP/tools/dist_train.bat fedvssl/conf/mmcv_conf/finetuning/r3d_18_ucf101/finetune_mob.py 1 --work_dir=./finetune_results --data_dir=fedvssl/data
```

After that, we perform the test process:

```bash
# Linux
bash fedvssl/CtP/tools/dist_test.sh fedvssl/conf/mmcv_conf/finetuning/r3d_18_ucf101/test_mob.py 1 --work_dir=./finetune_results --data_dir=fedvssl/data --progress

# or

# Windows
./fedvssl/CtP/tools/dist_test.bat fedvssl/conf/mmcv_conf/finetuning/r3d_18_ucf101/test_mob.py 1 --work_dir=./finetune_results --data_dir=fedvssl/data --progress
```