import os
import sys
import logging
import datetime
import glob
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import zipfile
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

#from metrics.mse import mse
from data.dataset import T4CDataset
from util.h5_util import write_data_to_h5
from model.configs import configs
from model.checkpointing import save_file_to_folder
from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY, MAX_FILE_DAY_IDX, TWO_HOURS

from uq.data_augmentation import DataAugmentation
from uq.aggregate import aggregate_tta
from metrics.pred_interval import get_quantile


def eval_model_tta(model: torch.nn.Module,
                   batch_size: int,
                   num_workers: int,
                   dataset_config: dict,
                   dataloader_config: dict,
                   model_str: str,
                   model_id: int,
                   save_checkpoint: str,
                   parallel_use: bool,
                   data_raw_path: str,
                   test_pred_path: str,
                   device: str,
                   dataset_limit: Optional[list] = None,
                   **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name

    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    post_transform = configs[model_str].get("post_transform", None)
    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]
    loss_sum = []

    if dataset_limit[0] is not None: # Limit #cities
        assert dataset_limit[0] <= len(cities)
        cities = cities[:dataset_limit[0]]

    logging.info(f"Evaluating '{model_str}' on '{device}' with '{loss_fct.__name__}' for {cities}.")

    for city in cities:
    
        test_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/test/*8ch.h5", recursive=True))
        logging.info(f"Test files extracted from {data_raw_path}/{city}/test/...")

        if test_pred_path is None:
            city_pred_path = Path(f"{data_raw_path}/{city}/test_pred_tta")
        else:
            city_pred_path = Path(os.path.join(test_pred_path, city))
        city_pred_path.mkdir(exist_ok=True, parents=True)

        if dataset_limit[1] is not None: # Limit #files per city
            assert dataset_limit[1] <= len(test_file_paths)
            test_file_paths = test_file_paths[:dataset_limit[1]]
        nr_files = len(test_file_paths)

        timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%m%d%H%M")
        model_pred_name = f"{model_str}_{city}_len{nr_files}_{timestamp}.zip"
        zip_file_path = city_pred_path / model_pred_name

        loss_city = []
        logging.info(f"Saving predictions with TTA on the {nr_files} " +
                     f"files for {city} as '{model_pred_name}'.")

        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            with tempfile.TemporaryDirectory() as tmpdir:
                for idx in range(nr_files):

                    file_filter = [Path(test_file_paths[idx])]
                    if (idx != len(test_file_paths) - 1): # Not last file idx
                        file_filter.append(Path(test_file_paths[idx + 1]))

                    samp_limit = dataset_limit[2] if dataset_limit[2] is not None else MAX_FILE_DAY_IDX

                    data = T4CDataset(root_dir=data_raw_path,
                                      file_filter=file_filter,
                                      dataset_limit=samp_limit + TWO_HOURS, # Limit #samples per file
                                      **dataset_config)

                    dataloader = DataLoader(dataset=data,
                                            batch_size=1, # Important to have 1
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=parallel_use,
                                            **dataloader_config)

                    pred, loss_file = evaluate(device=device,
                                                loss_fct=loss_fct,
                                                dataloader=dataloader,
                                                model=model,
                                                samp_limit=samp_limit,
                                                parallel_use=parallel_use,
                                                post_transform=post_transform)
                    loss_city.append(loss_file)

                    pred = aggregate_tta(pred) # Get aleatoric uncertainty

                    temp_h5 = os.path.join(tmpdir, f"pred_{city}_samp{idx}")
                    write_data_to_h5(data=pred, dtype=np.float16,
                                     filename=temp_h5, compression="lzf")
                    logging.info(f"Pred for file {idx+1}/{nr_files} written to .h5.")

                    arcname = str(file_filter[0]).split("/")[-1] # e.g. '2019-06-04_ANTWERP_8ch.h5'
                    zipf.write(filename=temp_h5, arcname=arcname)

        logging.info(f"Written all {nr_files} pred files for {city} to .zip.")
        zipf_mb_size = os.path.getsize(zip_file_path) / (1024 * 1024)
        logging.info(f"Zip file '{zip_file_path}' of size {zipf_mb_size:.1f} MB.")

        loss_sum.append(loss_city)
        logging.info("Loss for {}: {:.4f}".format(city, np.mean(loss_city)))

    logging.info("Loss over all cities: {:.4f}".format(np.mean(loss_sum)))
    folder_dir = os.path.join(save_checkpoint, f"{model_str}_{model_id}")
    comment = f"rows: loss per city, cols: loss per file for that city, {dataset_limit=}"
    save_file_to_folder(file=loss_sum, filename="loss_test", folder_dir=folder_dir,
                        fmt="%.4f", header=comment)


@torch.no_grad()
def evaluate(device, loss_fct, dataloader, model, samp_limit,
             parallel_use, post_transform, calib: bool = False) -> Tuple[torch.HalfTensor, float]:
    model.eval()
    loss_sum = 0
    data_augmenter = DataAugmentation()

    bsize = dataloader.batch_size
    if calib:
        batch_limit = samp_limit // bsize # No hard-coded limit
    else:
        batch_limit = min(MAX_FILE_DAY_IDX, samp_limit) // bsize # Only predict for batches up to at most idx 288

    pred = torch.empty( # Pred contains y_true + pred original img + pred for k augmented imgs: (samples, 1+1+k, 6, H, W, Ch)
        size=(batch_limit * bsize, 2 + data_augmenter.nr_augments, 6, 495, 436, 8),
        dtype=torch.float16, device=device)

    # Use only batch_size = 1 for dataloader since augmentations are interpreted as a batch
    with tqdm(dataloader) as tloader:
        for batch, (X, y) in enumerate(tloader):
            if batch == batch_limit:
                break

            X, y = X.to(device, non_blocking=parallel_use), y.to(device, non_blocking=parallel_use)

            X = X / 255
            X = data_augmenter.transform(X) # (1+k, 12 * Ch, H+pad, W+pad) in [0, 1]

            y_pred = model(X) # (1+k, 6 * Ch, H+pad, W+pad) in [0, 255]
            loss = loss_fct(y_pred[0, ...], y) # For original img

            y_pred = data_augmenter.detransform(y_pred) # (1+k, 6 * Ch, H+pad, W+pad)
            y_pred = post_transform(y_pred) # (1+k, 6, H, W, Ch]

            y_pred = torch.clamp(y_pred, 0, 255)
            y_pred = torch.cat((y, y_pred), dim=0) # add y_true: (1+1+k, 6, H, W, Ch)

            loss_sum += float(loss.item())
            loss_test = float(loss_sum/(batch+1))
            tloader.set_description(f"Batch {batch+1}/{batch_limit} > eval")
            tloader.set_postfix(loss = loss_test)

            assert pred[(batch * bsize):(batch * bsize + bsize)].shape == y_pred.shape
            pred[(batch * bsize):(batch * bsize + bsize)] = y_pred # Fill slice

    return pred, loss_test


def eval_calib_tta(model: torch.nn.Module,
                   num_workers: int,
                   dataset_config: dict,
                   dataloader_config: dict,
                   model_str: str,
                   parallel_use: bool,
                   data_raw_path: str,
                   device: str,
                   calibration_size: int = 500,
                   alpha: float = 0.1,
                   city_limit: Optional[int] = None,
                   to_file: bool = True,
                   **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name

    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    post_transform = configs[model_str].get("post_transform", None)
    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]

    if city_limit is not None: # Limit #cities
        assert city_limit <= len(cities)
        cities = cities[:city_limit]

    logging.info(f"Evaluating for {cities} with calibration sets of size {calibration_size}.")

    for city in cities:
    
        val_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/val/*8ch.h5", recursive=True))
        logging.info(f"Calibration files extracted from {data_raw_path}/{city}/val/...")

        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=val_file_paths,
                          **dataset_config)
        sub_idx = random.sample(range(0, len(data)), calibration_size)
        data = Subset(data, sub_idx)

        dataloader = DataLoader(dataset=data,
                                batch_size=1, # Important to have 1
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=parallel_use,
                                **dataloader_config)
        
        pred, _ = evaluate(device=device,
                           loss_fct=loss_fct,
                           dataloader=dataloader,
                           model=model,
                           samp_limit=calibration_size,
                           parallel_use=parallel_use,
                           post_transform=post_transform,
                           calib=True)

        pred = aggregate_tta(pred) # (samples, 3, 6, H, W, Ch)
        pred = get_quantile(pred, n=calibration_size, alpha=alpha) # (6, H, W, Ch)

        if to_file:
            file = os.path.join(data_raw_path, city, f"calib_quant_{int((1-alpha)*100)}_tta.h5")
            write_data_to_h5(data=pred, dtype=np.float16, filename=file, compression="lzf", verbose=True)

    logging.info(f"Written all calibration set {(1-alpha)*100}% quantiles for {cities=} to file.")

    del data, dataloader, pred


"""
def eval_calib_tta(model: torch.nn.Module,
                   batch_size: int,
                   num_workers: int,
                   dataset_config: dict,
                   dataloader_config: dict,
                   model_str: str,
                   parallel_use: bool,
                   data_raw_path: str,
                   device: str,
                   val_pred_path: str,
                   calibration_size: int = 500,
                   file_size: int = 100,
                   city_limit: Optional[int] = None,
                   **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name

    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    post_transform = configs[model_str].get("post_transform", None)
    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]

    if city_limit is not None: # Limit #cities
        assert city_limit <= len(cities)
        cities = cities[:city_limit]

    logging.info(f"Evaluating '{model_str}' on '{device}' with '{loss_fct.__name__}' for {cities}.")

    for city in cities:
    
        val_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/val/*8ch.h5", recursive=True))
        logging.info(f"Val files extracted from {data_raw_path}/{city}/val/...")

        if val_pred_path is None:
            city_pred_path = Path(f"{data_raw_path}/{city}/calib_pred_tta")
        else:
            city_pred_path = Path(os.path.join(val_pred_path, city))
        city_pred_path.mkdir(exist_ok=True, parents=True)

        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=val_file_paths,
                          **dataset_config)

        sub_idx = random.sample(range(0, len(data)), calibration_size)

        for i in range(0, calibration_size // file_size): # Manageable sample size for files

            data_file = Subset(data, sub_idx[(i * file_size):(i * file_size + file_size)])
    
            dataloader = DataLoader(dataset=data_file,
                                    batch_size=1, # Important to have 1!
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=parallel_use,
                                    **dataloader_config)
            
            pred, _ = evaluate(device=device,
                               loss_fct=loss_fct,
                               dataloader=dataloader,
                               model=model,
                               samp_limit=file_size,
                               parallel_use=parallel_use,
                               post_transform=post_transform)

            pred = aggregate_tta(pred) # Get aleatoric uncertainty

            file_path = os.path.join(city_pred_path, "calib_pred_tta_file" + str(i))
            write_data_to_h5(data=pred, dtype=np.float16, filename=file_path, compression="lzf", verbose=True)
            # (100, 3, 6, 495, 436, 8) float16 -> 57.8 GB file?

    logging.info(f"Written all TTA pred files for {cities=} for calibration set.")

    del data, data_file, dataloader, pred
"""
