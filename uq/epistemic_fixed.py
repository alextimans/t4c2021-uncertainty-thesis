import os
import sys
import logging
import glob
from pathlib import Path
from typing import Optional
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import T4CDataset
from model.configs import configs
from model.checkpointing import load_torch_model_from_checkpoint
from util.h5_util import write_data_to_h5
from util.logging import t4c_apply_basic_logging_config
from util.get_device import get_device
from util.set_seed import set_seed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_str", type=str, default="unet", required=False, choices=["unet"],
                        help="Model string name.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, required=False,
                        help="Path to torch model .pt checkpoint to be re-loaded.")
    parser.add_argument("--save_checkpoint", type=str, default="./checkpoints/", required=False,
                        help="Directory to store model checkpoints in.")
    parser.add_argument("--device", type=str, default=None, required=False, choices=["cpu", "cuda"],
                        help="Specify usage of specific device.")
    parser.add_argument("--random_seed", type=int, default=1234567, required=False,
                        help="Set manual random seed.")
    parser.add_argument("--data_parallel", type=str, default="False", required=False, choices=["True", "False"],
                        help="'Boolean' specifying use of DataParallel.")
    parser.add_argument("--num_workers", type=int, default=8, required=False,
                        help="Number of workers for data loader.")
    parser.add_argument("--batch_size", type=int, default=1, required=False,
                        help="Batch size for train, val and test data loaders. Preferably batch_size mod 2 = 0.")

    parser.add_argument("--data_raw_path", type=str, default="./data/raw", required=False,
                        help="Base directory of raw data.")
    parser.add_argument("--test_pred_path", type=str, default=None, required=False,
                        help="Specific directory to store test set model predictions in.")

    parser.add_argument("--uq_method", type=str, default=None, required=False, choices=["ensemble", "bnorm"],
                        help="Specify UQ method for epistemic uncertainty.")
    parser.add_argument("--fix_samp_idx", nargs=3, type=int, default=[None, None, None], required=False,
                        help="Fixed sample indices for time frame across training data per city (Bangkok, Barcelona, Moscow) in order.")

    return parser


def epistemic_hist(model: torch.nn.Module,
                cities: list,
                fix_samp_idx: list,
                batch_size: int,
                num_workers: int,
                dataset_config: dict,
                dataloader_config: dict,
                model_str: str,
                parallel_use: bool,
                data_raw_path: str,
                test_pred_path: str,
                device: str,
                uq_method: str,
                save_checkpoint: str,
                pred_to_file: bool = True,
                **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    
    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    uq_method_obj = configs[model_str]["uq_method"][uq_method] # Uncertainty object
    post_transform = configs[model_str]["post_transform"][uq_method]

    if uq_method == "ensemble":
        uq_method_obj.load_ensemble(device, save_checkpoint, configs[model_str]["model_class"], configs[model_str]["model_config"])
    elif uq_method == "bnorm":
        uq_method_obj.load_train_data(data_raw_path, configs[model_str]["dataset_config"]["point"])

    logging.info(f"Evaluating '{model_str}' on '{device}' for {cities} with {uq_method_obj.__class__}.")
    
    for i, city in enumerate(cities):
    
        train_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/train/*8ch.h5", recursive=True))
        logging.info(f"{len(train_file_paths)} train files extracted from {data_raw_path}/{city}/train/...")

        if test_pred_path is None:
            raise(AttributeError)
        else:
            res_path = Path(os.path.join(test_pred_path, city))
            res_path.mkdir(exist_ok=True, parents=True)
        
        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=train_file_paths,
                          **dataset_config)

        # idx of fixed sample index for each file
        logging.info(f"Using fixed sample index {fix_samp_idx[i]} for {city}.")
        sub_idx = [fix_samp_idx[i]+t*288 for t in range(len(train_file_paths))]

        dataloader = DataLoader(dataset=Subset(data, sub_idx),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=parallel_use,
                                **dataloader_config)

        pred, loss_city = uq_method_obj(device=device, # (samples, 3, H, W, Ch) torch.float32
                                       loss_fct=loss_fct,
                                       dataloader=dataloader,
                                       model=model,
                                       samp_limit=len(sub_idx),
                                       parallel_use=parallel_use,
                                       post_transform=post_transform)

        logging.info(f"Obtained predictions with uncertainty {uq_method} as {pred.shape, pred.dtype}.")
        if pred_to_file:
            write_data_to_h5(data=pred, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(res_path, f"pred_{uq_method}.h5"))

        logging.info(f"Evaluation via {uq_method} finished for {city}.")
    logging.info(f"Evaluation via {uq_method} finished for all cities in {cities}.")


def main():
    t4c_apply_basic_logging_config()
    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    parser = create_parser()
    args = parser.parse_args()

    # Named args (from parser + config file)
    model_str = args.model_str
    resume_checkpoint = args.resume_checkpoint
    device = args.device
    data_parallel = args.data_parallel
    random_seed = args.random_seed
    uq_method = args.uq_method

    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str]["model_config"]
    dataset_config = configs[model_str]["dataset_config"][uq_method]
    dataloader_config = configs[model_str]["dataloader_config"]

    # Set (all) seeds
    random_seed = set_seed(random_seed)
    logging.info(f"Used {random_seed=} for seeds.")

    # Model setup
    model = model_class(**model_config)
    assert model_class == model.__class__, f"{model.__class__=} invalid."
    logging.info(f"Created model of class {model_class}.")

    # Device setting
    device, parallel_use = get_device(device, data_parallel)
    if parallel_use: # Multiple GPU usage
        model = torch.nn.DataParallel(model)
        logging.info(f"Using {len(model.device_ids)} GPUs: {model.device_ids}.")
        device = f"cuda:{model.device_ids[0]}" # cuda:0 is main process device
    logging.info(f"Using {device=}, {parallel_use=}.")
    vars(args).pop("device")
     
    # Checkpoint loading
    if resume_checkpoint is not None:
        load_torch_model_from_checkpoint(checkpt_path=resume_checkpoint,
                                         model=model, map_location=device)
    else:
        logging.info("No model checkpoint given.")

    epistemic_hist(model=model,
                   cities=["BANGKOK", "BARCELONA", "MOSCOW"],
                   dataset_config=dataset_config,
                   dataloader_config=dataloader_config,
                   parallel_use=parallel_use,
                   device=device,
                   **(vars(args)))
    logging.info("Main finished.")


if __name__ == "__main__":
    main()
