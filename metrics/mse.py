
import torch
from torch.nn.functional import mse_loss


def mse(pred) -> float:

   """
   Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim
   '2' is y_true (0) and some prediction (1) (e.g. only point or point + uncertainty).
   Trailing dimensions (6, H, W, Ch) may be of arbitrary sizes.
   Returns: MSE over all dimensions.
   """

   return mse_loss(pred[:, 1, ...], target=pred[:, 0, ...], reduction="mean")


def mse_samples(pred) -> float:

    """
    Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is y_true (0) and some prediction (1) (e.g. only point or point + uncertainty).
    Returns: MSE over the sample dimension as tensor (6, H, W, Ch).
    """

    return torch.sqrt(torch.mean((pred[:, 0, ...] - pred[:, 1, ...])**2, dim=0))
