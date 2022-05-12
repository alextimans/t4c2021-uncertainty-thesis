from functools import partial

import torch
import torchvision.transforms.functional as TF

class DataAugmentation:
    def __init__(self):
        self.transformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=90),
            partial(TF.rotate, angle=180),
            partial(TF.rotate, angle=270)
            ]

        self.detransformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=-90),
            partial(TF.rotate, angle=-180),
            partial(TF.rotate, angle=-270)
            ]

        self.nr_augments = len(self.transformations)

    def transform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives X = (1, 12*8, 496, 448) and does k augmentations 
        returning X' = (1+k, 12*8, 496, 448)
        """

        X = data
        for transform in self.transformations:
            X_aug = transform(data)
            X = torch.cat((X, X_aug), dim=0)

        assert list(X.shape) == [self.nr_augments+1] + list(data.shape[1:])

        return X

    def detransform(self, data: torch.Tensor) -> torch.Tensor:

        """
        receives y_pred = (1+k, 6*8, 496, 448), detransforms the 
        k augmentations and returns y_pred = (1+k, 6*8, 496, 448)
        """

        y = data[0, ...].unsqueeze(dim=0)
        for i, detransform in enumerate(self.detransformations):
            y_deaug = detransform(data[i+1, ...].unsqueeze(dim=0))
            y = torch.cat((y, y_deaug), dim=0)

        assert y.shape == data.shape

        return y
