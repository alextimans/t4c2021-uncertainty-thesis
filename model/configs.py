# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

from functools import partial
from model.unet import UNet
from model.unet import UNetTransformer


configs = {
    "unet": {
        "model_class": UNet,

        "model_config": {
            "in_channels": 12 * 8,
            "out_channels": 6 * 8,
            "depth": 5,
            "layer_out_pow2": 6, # t4c arg: 'wf'
            "padding": True,
            "batch_norm": True,
            "up_mode": "upconv"
            },

        "dataset_config": {
            "transform": partial(UNetTransformer.unet_pre_transform,
                                 stack_channels_on_time=True,
                                 zeropad2d=(6, 6, 1, 0),
                                 batch_dim=False)
            },

        "pre_transform": partial(UNetTransformer.unet_pre_transform,
                                 stack_channels_on_time=True,
                                 zeropad2d=(6, 6, 1, 0),
                                 batch_dim=True,
                                 from_numpy=False),

        "post_transform": partial(UNetTransformer.unet_post_transform,
                                  unstack_channels_on_time=True,
                                  crop=(6, 6, 1, 0),
                                  batch_dim=True),

        "dataloader_config": {
            },

        "optimizer_config": { # Default params for Adam
            "lr": 1e-4, # default Adam: 1e-3
            "betas": (0.9, 0.999),
            "weight_decay": 0,
            "amsgrad": False,
            },

        "lr_scheduler_config": {
            "patience": 0, # effect after epoch patience+1 without improvement
            "mode": "min",
            "factor": 0.1,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "min_lr": 1e-5, # max. 2 scheduler steps in relation to lr
            "verbose": True
            },

        "earlystop_config": {
            "patience": 1, # effect after epoch patience+1 without improvement
            "delta": 1e-4,
            "save_each_epoch": True,
            "loss_improve": "min",
            "verbose": True
            }
    }
}
