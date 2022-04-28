# base code from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

import logging
import numpy as np
import torch

from model.checkpointing import save_torch_model_to_checkpoint


class EarlyStopping:
    def __init__(self,
                 patience: int = 3,
                 delta: float = 0,
                 verbose: bool = False,
                 save_each_epoch: bool = False,
                 loss_improve: str = "min"):

        """
        Early stops model training if val loss doesn't improve after a given patience.

        Parameters
        ----------
        patience: int
            How long to wait after the last time val loss improved before stopping.
        delta: float
            Minimum change in the monitored loss to qualify as an improvement.
        verbose: bool
            If True prints a message for each validation loss improvement.
        save_each_epoch: bool
            Should the model be saved each epoch regardless of improvement or not.
        loss_improve: str
            Loss function-specific improvement direction. One in ["min", "max"].       
        """

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_each_epoch = save_each_epoch
        self.loss_improve = loss_improve

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.loss_val_min = -np.Inf if self.loss_improve == "max" else np.Inf

    def __call__(self,
                 model: torch.nn.Module,
                 loss_val: float = None,
                 epoch: int = None,
                 model_str: str = None,
                 model_id: int = None,
                 save_checkpoint: str = None):

        loss = loss_val
        if (self.loss_improve == "max"):
            loss = -loss_val
            self.delta = -self.delta

        if self.best_loss is None: # First call
            self.best_loss = loss
            self._save_checkpoint(model, loss_val, epoch, model_str, model_id, save_checkpoint)

        elif (loss >= self.best_loss - self.delta): # No improvement in val loss
            self.counter += 1
            logging.info(f"EarlyStopping being patient: {self.counter}/{self.patience}.")

            if (self.counter > self.patience): # Init early stopping
                self.early_stop = True

            if self.save_each_epoch:
                self._save_checkpoint(model, loss_val, epoch, model_str, model_id, save_checkpoint)

        else: # Improvement in val loss
            self.best_loss = loss
            self._save_checkpoint(model, loss_val, epoch, model_str, model_id, save_checkpoint)
            self.counter = 0

    def _save_checkpoint(self, model, loss_val, epoch, model_str, model_id, save_checkpoint):
        
        """
        Saves model to checkpoint and optionally displays loss change.
        """

        if self.verbose:
            logging.info(f"Val loss change: {self.loss_val_min:.4f} -> {loss_val:.4f}.")
        save_torch_model_to_checkpoint(model=model,
                                       model_str=model_str,
                                       model_id=model_id,
                                       epoch=epoch,
                                       save_checkpoint=save_checkpoint)
        self.loss_val_min = loss_val
