"""Config file used for fine-tuning on MOB dataset."""

_base_ = ["../model_r3d18.py", "../runtime_mob.py"]

work_dir = "./finetune_results/"

model = {
    "backbone": {
        "pretrained": "finetune_results/epoch_5.pth",
    },
}
