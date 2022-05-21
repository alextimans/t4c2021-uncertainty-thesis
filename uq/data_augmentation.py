from functools import partial

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as TF


class DataAugmentation:
    def __init__(self):
        self.transformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=90, expand=True),
            partial(TF.rotate, angle=180, expand=True),
            partial(TF.rotate, angle=270, expand=True),
            tf.Compose([TF.vflip, partial(TF.rotate, angle=90, expand=True)]),
            tf.Compose([TF.vflip, partial(TF.rotate, angle=-90, expand=True)])
            ]

        self.detransformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=-90, expand=True),
            partial(TF.rotate, angle=-180, expand=True),
            partial(TF.rotate, angle=-270, expand=True),
            tf.Compose([partial(TF.rotate, angle=-90, expand=True), TF.vflip]),
            tf.Compose([partial(TF.rotate, angle=90, expand=True), TF.vflip])
            ]

        self.nr_augments = len(self.transformations)

    def transform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives X = (1, 12 * Ch, H, W) and does k augmentations 
        returning X' = (1+k, 12 * Ch, H, W).
        """

        X = data
        for transform in self.transformations:
            X_aug = transform(data)
            X = torch.cat((X, X_aug), dim=0)

        assert list(X.shape) == [1+self.nr_augments] + list(data.shape[1:])

        return X

    def detransform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives y_pred = (1+k, 6 * Ch, H, W), detransforms the 
        k augmentations and returns y_pred = (1+k, 6 * Ch, H, W).
        """

        y = data[0, ...].unsqueeze(dim=0)
        for i, detransform in enumerate(self.detransformations):
            y_deaug = detransform(data[i+1, ...].unsqueeze(dim=0))
            y = torch.cat((y, y_deaug), dim=0)

        assert y.shape == data.shape

        return y
