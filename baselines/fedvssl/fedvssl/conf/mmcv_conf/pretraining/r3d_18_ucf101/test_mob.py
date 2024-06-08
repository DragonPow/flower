"""Config file used for pre-training on MOB dataset."""

_base_ = ["../model_r3d18.py", "../pretraining_runtime_mob.py"]

work_dir = "./fedvssl_results/"

model = {
    "backbone": {
        "pretrained": "fedvssl_results/epoch_1.pth",
    },
}
