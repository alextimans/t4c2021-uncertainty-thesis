# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import os
import datetime
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch


def load_torch_model_from_checkpoint(checkpt_path: Union[str, Path],
                                     model: torch.nn.Module,
                                     map_location: str = None):

    state_dict = torch.load(checkpt_path, map_location)
    assert isinstance(state_dict, dict) and ("model" in state_dict)

    state_model = state_dict["model"]
    parallel_checkpt = all("module." in key for key in list(state_model.keys()))

    if not isinstance(model, torch.nn.DataParallel) and parallel_checkpt:
        new_state_model = state_model.copy()
        for key, val in state_model.items(): # remove "module." for successful match
            new_state_model[key[7:]] = new_state_model.pop(key)
        state_model = new_state_model
        logging.info("Mismatch model <-> state_dict, removed 'module.' from keys.")

    elif isinstance(model, torch.nn.DataParallel) and not parallel_checkpt:
        new_state_model = state_model.copy()
        for key, val in state_model.items(): # add "module." for successful match
            new_state_model["module." + key] = new_state_model.pop(key)
        state_model = new_state_model
        logging.info("Mismatch model <-> state_dict, added 'module.' to keys.")

    model.load_state_dict(state_model)
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
