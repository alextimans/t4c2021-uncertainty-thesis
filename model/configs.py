# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

from functools import partial
from model.unet import UNet
from model.unet import UNetTransformer
from uq import point_pred, data_augmentation, ensemble, stochastic_batchnorm


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

        "uq_method": {
            "point": point_pred.PointPred(),
            "tta": data_augmentation.DataAugmentation(),
            "ensemble": ensemble.DeepEnsemble(load_from_epoch=[1,1,1,1,1]),
            "bnorm": stochastic_batchnorm.StochasticBatchNorm(passes=2, train_batch_size=2)
        },

        "dataset_config": {
            "point": {
                "transform": partial(UNetTransformer.unet_pre_transform,
                                     stack_channels_on_time=True,
                                     zeropad2d=(6, 6, 1, 0), # (495, 436) -> (496, 448)
                                     batch_dim=False)
                },
            "tta": {
                "transform": partial(UNetTransformer.unet_pre_transform,
                                     stack_channels_on_time=True,
                                     zeropad2d=(30, 30, 1, 0), # (495, 436) -> (496, 496)
                                     batch_dim=False)
                },
            "ensemble": {
                "transform": partial(UNetTransformer.unet_pre_transform,
                                     stack_channels_on_time=True,
                                     zeropad2d=(6, 6, 1, 0), # (495, 436) -> (496, 448)
                                     batch_dim=False)
                },
            "bnorm": {
                "transform": partial(UNetTransformer.unet_pre_transform,
                                     stack_channels_on_time=True,
                                     zeropad2d=(6, 6, 1, 0), # (495, 436) -> (496, 448)
                                     batch_dim=False)
                }
            },

        "pre_transform": {
            "point": partial(UNetTransformer.unet_pre_transform,
                                 stack_channels_on_time=True,
                                 zeropad2d=(6, 6, 1, 0),
                                 batch_dim=True,
                                 from_numpy=False),
            "tta": partial(UNetTransformer.unet_pre_transform,
                                 stack_channels_on_time=True,
                                 zeropad2d=(30, 30, 1, 0),
                                 batch_dim=True,
                                 from_numpy=False),
            "ensemble": partial(UNetTransformer.unet_pre_transform,
                                 stack_channels_on_time=True,
                                 zeropad2d=(6, 6, 1, 0),
                                 batch_dim=True,
                                 from_numpy=False),
            "bnorm": partial(UNetTransformer.unet_pre_transform,
                                 stack_channels_on_time=True,
                                 zeropad2d=(6, 6, 1, 0),
                                 batch_dim=True,
                                 from_numpy=False)
            },

        "post_transform": {
            "point": partial(UNetTransformer.unet_post_transform,
                                  unstack_channels_on_time=True,
                                  crop=(6, 6, 1, 0),
                                  batch_dim=True),
            "tta": partial(UNetTransformer.unet_post_transform,
                                  unstack_channels_on_time=True,
                                  crop=(30, 30, 1, 0),
                                  batch_dim=True),
            "ensemble": partial(UNetTransformer.unet_post_transform,
                                  unstack_channels_on_time=True,
                                  crop=(6, 6, 1, 0),
                                  batch_dim=True),
            "bnorm": partial(UNetTransformer.unet_post_transform,
                                  unstack_channels_on_time=True,
                                  crop=(6, 6, 1, 0),
                                  batch_dim=True),
            },

        "dataloader_config": {
            },

        "optimizer_config": { # Default params for Adam
            "lr": 1e-4, # default Adam: 1e-3
            "betas": (0.9, 0.999),
            "weight_decay": 0,
            "amsgrad": False,
            },

        "lr_scheduler_config": {
            "patience": 2, # effect after epoch patience+1 without improvement
            "mode": "min",
            "factor": 0.1,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "min_lr": 1e-6, # max. 2 scheduler steps in relation to lr
            "verbose": True
            },

        "earlystop_config": {
            "patience": 3, # effect after epoch patience+1 without improvement
            "delta": 1e-4,
            "save_each_epoch": True,
            "loss_improve": "min",
            "verbose": True
            }
    }
}
