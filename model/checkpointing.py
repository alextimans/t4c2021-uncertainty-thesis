# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import os
import datetime
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.nn import DataParallel

from util.get_device import get_device


def load_torch_model_from_checkpoint(checkpt_path: Union[str, Path],
                                     model: torch.nn.Module,
                                     map_location: str = None) -> torch.nn.Module:

    device, _ = get_device(map_location)
    state_dict = torch.load(checkpt_path, map_location=device)

    if isinstance(state_dict, DataParallel):
        logging.info("state_dict instance of DataParallel.")
        state_dict = state_dict.state_dict()
        new_state_dict = OrderedDict()

        for key, val in state_dict.items():
            if key[:7] == "module.":
                key = key[7:]  # remove "module." if trained with data parallelism
            new_state_dict[key] = val

        state_dict = new_state_dict

    elif isinstance(state_dict, dict) and ("model" in state_dict): # Normal use case
        logging.info("state_dict is model attribute.")
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    logging.info(f"Loaded model from checkpoint '{checkpt_path}'.")


def save_torch_model_to_checkpoint(model: torch.nn.Module, model_str: str,
                                   model_id: int, epoch: int, save_checkpoint: str = ""):

    """ 
    Saves a torch model as a checkpoint in specified location.

    Parameters
    ----------
    model: torch.nn.Module
        Model to create checkpoint of.
    model_str: str
        Model string name.
    model_id: int
        Model ID to create unique checkpoints folder.
    epoch: int
        Nr. of epochs model was trained.
    save_checkpoint: str
        Path to checkpoints folder. Default is local directory.
    """

    checkpt_path = Path(os.path.join(save_checkpoint, f"{model_str}_{model_id}"))
    checkpt_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%m%d%H%M")
    save_dict = {"epoch": epoch, "model": model.state_dict()}
    checkpt_name = f"{model_str}_ep{epoch+1}_{timestamp}.pt"
    path = os.path.join(checkpt_path, checkpt_name)

    torch.save(save_dict, path)
    logging.info(f"Model {model_str} trained for {epoch+1} epochs saved as '{path}'.")


def save_file_to_folder(file = None, filename: str = None,
                        folder_dir: Union[Path, str] = None, **kwargs):

    """ 
    Stores file in specified folder as .txt file.
    """

    folder_path = Path(folder_dir) if isinstance(folder_dir, str) else folder_dir
    folder_path.mkdir(exist_ok=True, parents=True)

    np.savetxt(os.path.join(folder_path, f"{filename}.txt"), file, **kwargs)
    logging.info(f"Written {filename}.txt to {folder_path}.")


"""
def save_loss_to_checkpoint(model_str: str, model_id: int, save_checkpoint: str = "",
                            loss_train: list = None, loss_val: list = None):

    #Stores lists loss_train, loss_val containing losses
    #in the current model checkpoint folder as .txt files.

    checkpt_path = Path(os.path.join(save_checkpoint, f"{model_str}_{model_id}"))
    checkpt_path.mkdir(exist_ok=True, parents=True)

    np.savetxt(os.path.join(checkpt_path, "loss_train.txt"), loss_train, fmt="%.4f")
    np.savetxt(os.path.join(checkpt_path, "loss_val.txt"), loss_val, fmt="%.4f")
    logging.info("Written loss_train.txt and loss_val.txt to checkpoint folder.")
"""
