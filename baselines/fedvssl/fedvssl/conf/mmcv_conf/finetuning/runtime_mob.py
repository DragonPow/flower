"""Config file used for fine-tuning on MOB dataset."""

dist_params = {"backend": "nccl"}
log_level = "INFO"
load_from = None
resume_from = None
syncbn = True

train_cfg = None
test_cfg = None
evaluation = {"interval": 5}

data = {
    "videos_per_gpu": 4,  # total batch size 8*4 == 32
    "workers_per_gpu": 4,
    "train": {
        "type": "TSNDataset",
        "name": "mob_train_split",
        "data_source": {
            "type": "JsonClsDataSource",
            "ann_file": "mob/annotations/finetuning.json",
        },
        "backend": {
            "type": "ZipBackend",
            "zip_fmt": "mob/zips/{}.zip",
            "frame_fmt": "img_{:05d}.jpg",
        },
        "frame_sampler": {
            "type": "RandomFrameSampler",
            "num_clips": 1,
            "clip_len": 16,
            "strides": 2,
            "temporal_jitter": False,
        },
        "test_mode": False,
        "transform_cfg": [
            {"type": "GroupScale", "scales": [(149, 112), (171, 128), (192, 144)]},
            {"type": "GroupFlip", "flip_prob": 0.35},
            {"type": "RandomBrightness", "prob": 0.20, "delta": 32},
            {"type": "RandomContrast", "prob": 0.20, "delta": 0.20},
            {
                "type": "RandomHueSaturation",
                "prob": 0.20,
                "hue_delta": 12,
                "saturation_delta": 0.1,
            },
            {"type": "GroupRandomCrop", "out_size": 112},
            {
                "type": "GroupToTensor",
                "switch_rgb_channels": True,
                "div255": True,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        ],
    },
    "val": {
        "type": "TSNDataset",
        "name": "mob_test_split2",
        "data_source": {
            "type": "JsonClsDataSource",
            "ann_file": "mob/annotations/test_split_2.json",
        },
        "backend": {
            "type": "ZipBackend",
            "zip_fmt": "mob/zips/{}.zip",
            "frame_fmt": "img_{:05d}.jpg",
        },
        "frame_sampler": {
            "type": "UniformFrameSampler",
            "num_clips": 10,
            "clip_len": 16,
            "strides": 2,
            "temporal_jitter": False,
        },
        "test_mode": True,
        "transform_cfg": [
            {"type": "GroupScale", "scales": [(171, 128)]},
            {"type": "GroupCenterCrop", "out_size": 112},
            {
                "type": "GroupToTensor",
                "switch_rgb_channels": True,
                "div255": True,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        ],
    },
    "test": {
        "type": "TSNDataset",
        "name": "mob_test_split2",
        "data_source": {
            "type": "JsonClsDataSource",
            "ann_file": "mob/annotations/test_split_2.json",
        },
        "backend": {
            "type": "ZipBackend",
            "zip_fmt": "mob/zips/{}.zip",
            "frame_fmt": "img_{:05d}.jpg",
        },
        "frame_sampler": {
            "type": "UniformFrameSampler",
            "num_clips": 10,
            "clip_len": 16,
            "strides": 2,
            "temporal_jitter": False,
        },
        "test_mode": True,
        "transform_cfg": [
            {"type": "GroupScale", "scales": [(171, 128)]},
            {"type": "GroupCenterCrop", "out_size": 112},
            {
                "type": "GroupToTensor",
                "switch_rgb_channels": True,
                "div255": True,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
        ],
    },
}

# optimizer
total_epochs = 5
optimizer = {"type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4}
optimizer_config = {"grad_clip": {"max_norm": 40, "norm_type": 2}}
# learning policy
lr_config = {"policy": "step", "step": [5, 10, 13], "gamma": 0.5}
checkpoint_config = {"interval": 1, "max_keep_ckpts": 1, "create_symlink": False}
workflow = [("train", 8)]
log_config = {
    "interval": 5,
    "hooks": [
        {"type": "TextLoggerHook"},
        {"type": "TensorboardLoggerHook"},
    ],
}
