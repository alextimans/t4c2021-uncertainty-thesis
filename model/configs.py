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
            "layer_out_pow2": 6,
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

        "optimizer_config": {
            "lr": 1e-4
            },

        "training_config": { # Early stopping
            "patience": 3,
            "delta": 0,
            "save_each_epoch": False,
            "loss_improve": "min"
            }
    }
}
