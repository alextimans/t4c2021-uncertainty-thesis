# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import MSELoss
from torch.nn.functional import mse_loss

from data.data_layout import volume_channel_indices, speed_channel_indices


def mse(y: Union[torch.Tensor, np.ndarray],
        y_pred: Union[torch.Tensor, np.ndarray],
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        mask_norm: bool = True,
        axis: Optional[Tuple] = None,
        indices: Optional[List] = None,
        use_np: bool = False):

    if (axis is not None) or use_np:
        loss = _np_mse(y, y_pred, mask, mask_norm, axis, indices)
    else:
        loss = _torch_mse(y, y_pred, mask, mask_norm, indices)

    return loss


def _torch_mse(y: Union[torch.Tensor, np.ndarray],
               y_pred: Union[torch.Tensor, np.ndarray],
               mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
               mask_norm: bool = True,
               indices: Optional[List] = None):

    if (type(y) != torch.Tensor) or (type(y_pred) != torch.Tensor):
        y_i = torch.from_numpy(y[:]).float()
        y_pred_i = torch.from_numpy(y_pred[:]).float()
    else:
        y_i = y[:]
        y_pred_i = y_pred[:]

    if indices is not None:
        y_i = y_i[..., indices]
        y_pred_i = y_pred_i[..., indices]

    if mask is not None:
        if (type(mask) != torch.Tensor):
            mask_i = torch.from_numpy(mask[:]).float()
        else:
            mask_i = mask[:]

        if indices is not None:
            mask_i = mask_i[..., indices]

        y_i = y_i * mask_i
        y_pred_i = y_pred_i * mask_i

        if mask_norm:
            mask_ratio = np.count_nonzero(mask) / mask.size

            return mse_loss(y_pred_i, y_i).numpy() / mask_ratio

    return mse_loss(y_pred_i, y_i).numpy()


def _np_mse(y: Union[torch.Tensor, np.ndarray],
            y_pred: Union[torch.Tensor, np.ndarray],
            mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
            mask_norm: bool = True,
            axis: Optional[Tuple] = None,
            indices: Optional[List] = None):

    if indices is not None:
        y = y[..., indices]
        y_pred = y_pred[..., indices]

    if mask is not None:
        y = y * mask
        y_pred = y_pred * mask

    y_i = y.astype(np.int64)
    y_pred_i = y_pred.astype(np.int64)

    if mask is not None and mask_norm:
        mask_ratio = np.count_nonzero(mask) / mask.size

        return (np.square(np.subtract(y_i, y_pred_i))).mean(axis=axis) / mask_ratio

    return (np.square(np.subtract(y_i, y_pred_i))).mean(axis=axis)


class MSELossWiedemann(MSELoss):
    def __init__(self,
                 size_average = None,
                 reduce = None,
                 reduction: str = "mean"):

        super(MSELossWiedemann, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = mse_loss_wiedemann(input, target, reduction=self.reduction)

        return loss


def mse_loss_wiedemann(input: torch.Tensor,
                       target: torch.Tensor,
                       reduction: str = "mean") -> torch.Tensor:

    VOL_CHANNELS = volume_channel_indices
    SPEED_CHANNELS = speed_channel_indices

    n = (torch.count_nonzero(target[..., VOL_CHANNELS] != 0)
         + torch.count_nonzero(target[..., VOL_CHANNELS] == 0))
    f = torch.count_nonzero(target[..., VOL_CHANNELS] != 0) + n

    mask = ((target[..., VOL_CHANNELS] != 0)).float()
    target[..., SPEED_CHANNELS] = target[..., SPEED_CHANNELS] * mask
    input[..., SPEED_CHANNELS] = input[..., SPEED_CHANNELS] * mask
    masked_loss = mse_loss(input, target, reduction=reduction) / (f * 2 * n)

    return masked_loss


# The torch mse is significantly faster than (np.square(np.subtract(y, y_pred))).mean(axis=axis)
# Results from performance comparison:
# %timeit mse(prediction, ground_truth_prediction)
# 17.9 ms ± 168 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# %timeit mse(prediction, ground_truth_prediction, use_np=True)
# 70.1 ms ± 1.93 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# %timeit mse(prediction, ground_truth_prediction, mask=static_mask)
# 52.4 ms ± 1.15 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# %timeit mse(prediction, ground_truth_prediction, mask=static_mask, use_np=True)
# 112 ms ± 1.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
