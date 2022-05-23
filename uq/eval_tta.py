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

from data.dataset import T4CDataset
from util.h5_util import write_data_to_h5, load_h5_file
from model.configs import configs
from model.checkpointing import save_file_to_folder
from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY, MAX_FILE_DAY_IDX, TWO_HOURS

from uq.data_augmentation import DataAugmentation
from uq.aggregate import aggregate_tta

from metrics.pred_interval import get_quantile, get_pred_interval, coverage, mean_pi_width
from metrics.calibration import ence, coeff_variation
from metrics.mse import mse, mse_samples


def eval_model_tta(model: torch.nn.Module,
                   batch_size: int,
                   num_workers: int,
                   dataset_config: dict,
                   dataloader_config: dict,
                   model_str: str,
                   model_id: int,
                   parallel_use: bool,
                   data_raw_path: str,
                   test_pred_path: str,
                   device: str,
                   uq_method: str,
                   quantiles_path: str,
                   sample_idx_path: str = None,
                   dataset_limit: Optional[list] = None,
                   pred_to_file: bool = False,
                   pi_to_file: bool = False,
                   scores_to_file: bool = True,
                   **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    
    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    post_transform = configs[model_str]["post_transform"][uq_method]
    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]
    
    if dataset_limit[0] is not None: # Limit #cities
        assert dataset_limit[0] <= len(cities)
        cities = cities[:dataset_limit[0]]
    
    logging.info(f"Evaluating '{model_str}' on '{device}' for {cities} with {uq_method=}.")
    
    for city in cities:
    
        test_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/test/*8ch.h5", recursive=True))
        logging.info(f"Test files extracted from {data_raw_path}/{city}/test/...")

        if test_pred_path is None:
            city_pred_path = Path(f"{data_raw_path}/{city}/test_{uq_method}_{model_str+str(model_id)}")
        else:
            city_pred_path = Path(os.path.join(test_pred_path, city))
        city_pred_path.mkdir(exist_ok=True, parents=True)
        
        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=test_file_paths,
                          **dataset_config)
        
        if sample_idx_path is not None:
            sub_idx = torch.from_numpy(np.loadtxt(os.path.join(data_raw_path, city, sample_idx_path))).to(torch.int)
        else:
            sub_idx = range(0, len(data))
        logging.info(f"Evaluating for {city} on {len(sub_idx)} test set indices in range {torch.min(sub_idx).item(), torch.max(sub_idx).item()}.")
        
        data = Subset(data, sub_idx)
        
        dataloader = DataLoader(dataset=data,
                                batch_size=1, # Important to have 1
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=parallel_use,
                                **dataloader_config)
        
        pred, loss_city = evaluate(device=device,
                                   loss_fct=loss_fct,
                                   dataloader=dataloader,
                                   model=model,
                                   samp_limit=len(sub_idx),
                                   parallel_use=parallel_use,
                                   post_transform=post_transform,
                                   no_day_limit=True)
        logging.info(f"Obtained predictions as {pred.shape, pred.dtype}.")

        pred = aggregate_tta(pred) # (samples, 3, 6, H, W, Ch)
        logging.info(f"Aggregated predictions to obtain uncertainty via {uq_method} as {pred.shape, pred.dtype}.")
        if pred_to_file:
            write_data_to_h5(data=pred, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(city_pred_path, f"pred_{uq_method}.h5"))
        
        quant = os.path.join(data_raw_path, city, quantiles_path)
        pred_interval = get_pred_interval(pred[:, 1:, ...], load_h5_file(quant, dtype=torch.float16)) # (samples, 2, 6, H, W, Ch)
        logging.info(f"Obtained prediction intervals as {pred_interval.shape, pred_interval.dtype}.")
        if pi_to_file:
            write_data_to_h5(data=pred_interval, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(city_pred_path, f"pi_{uq_method}.h5"))
        
        # tensor (5, 6, H, W, Ch) containing all the metric scores across the sample dimension
        scores = torch.stack((mean_pi_width(pred_interval.to(torch.float32)),
                              coverage(torch.cat((pred[:, 0, ...].unsqueeze(dim=1), pred_interval), dim=1).to(torch.float32)),
                              ence(pred.to(torch.float32)),
                              coeff_variation(pred[:, 2, ...].to(torch.float32)),
                              mse_samples(pred[:, :2, ...].to(torch.float32))
                              ), dim=0)
        logging.info(f"Obtained metric scores across sample dimension as {scores.shape, scores.dtype}.")
        del data, pred, pred_interval
        
        if scores_to_file:
            write_data_to_h5(data=scores, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(city_pred_path, f"scores_{uq_method}.h5"))
        
        horizon, channels = 5, [1, 3, 5, 7] # pred 1h into future + speed channels only
        scalar_scores = torch.Tensor((
            torch.mean(scores[0, horizon, :, :, channels]),
            torch.mean(scores[1, horizon, :, :, channels]),
            torch.mean(scores[2, horizon, :, :, channels]),
            torch.mean(scores[3, horizon, :, :, channels]),
            torch.mean(scores[4, horizon, :, :, channels])))
        del scores

        logging.info(f"Scalar scores for prediction {horizon=} and {channels=}:")
        logging.info(f"[mean PI width, cov, ence, coeff_var, mse] --> {scalar_scores.numpy()}.")
        save_file_to_folder(file=scalar_scores, filename=f"scalar_scores_{uq_method}", folder_dir=city_pred_path,
                            fmt="%.4f", header="[mean PI width, cov, ence, coeff_var, mse]")

        logging.info(f"Evaluation via {uq_method} finished for {city}.")
    logging.info(f"Evaluation via {uq_method} finished for all cities in {cities}.")

# pred: float16 / pred_interval: float16 / scores: float32 / scalar_scores: float32
# mse_samples: (samples, 2, 6, H, W, Ch) -> (6, H, W, Ch)
# ence: (samples, 3, 6, H, W, Ch) -> (6, H, W, Ch)
# coeff_variation: (samples, 6, H, W, Ch) -> (6, H, W, Ch)
# get_pred_interval: (samples, 2, 6, H, W, Ch) + (6, H, W, Ch) -> (samples, 2, 6, H, W, Ch)
# coverage: (samples, 3, 6, H, W, Ch) -> (6, H, W, Ch)
# mean_pi_width: (samples, 2, 6, H, W, Ch) -> (6, H, W, Ch)

@torch.no_grad()
def evaluate(device, loss_fct, dataloader, model, samp_limit,
             parallel_use, post_transform, no_day_limit: bool = False) -> Tuple[torch.HalfTensor, float]:
    model.eval()
    loss_sum = 0
    data_augmenter = DataAugmentation()

    bsize = dataloader.batch_size
    if no_day_limit:
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
            # logging.info(f"{X.shape, torch.min(X), torch.max(X)}")
            # logging.info(f"{y.shape, torch.min(y), torch.max(y)}")
            X = X / 255
            # logging.info(f"{X.shape, torch.min(X), torch.max(X)}")
            X = data_augmenter.transform(X) # (1+k, 12 * Ch, H+pad, W+pad) in [0, 1]
            # logging.info(f"{X.shape, torch.min(X), torch.max(X)}")

            y_pred = model(X) # (1+k, 6 * Ch, H+pad, W+pad) in [0, 255]
            # logging.info(f"{y_pred.shape, torch.min(y_pred), torch.max(y_pred)}")
            loss = loss_fct(y_pred[0, :, 1:, 30:-30], y[:, :, 1:, 30:-30].squeeze(dim=0)) # For original img & unpadded

            y_pred = data_augmenter.detransform(y_pred) # (1+k, 6 * Ch, H+pad, W+pad)
            # logging.info(y_pred.shape)
            y_pred = post_transform(y_pred) # (1+k, 6, H, W, Ch)
            # logging.info(y_pred.shape)
            y_pred = torch.cat((post_transform(y), y_pred), dim=0) # add y_true: (1+1+k, 6, H, W, Ch)
            # logging.info(f"{y_pred.shape, torch.min(y_pred), torch.max(y_pred)}")
            y_pred = torch.clamp(y_pred, 0, 255).unsqueeze(dim=0)
            # logging.info(f"{y_pred.shape, torch.min(y_pred), torch.max(y_pred)}")

            loss_sum += float(loss.item())
            loss_test = float(loss_sum/(batch+1))
            tloader.set_description(f"Batch {batch+1}/{batch_limit} > eval")
            tloader.set_postfix(loss = loss_test)

            # logging.info(pred.shape)
            # logging.info(pred[(batch * bsize):(batch * bsize + bsize)].shape)
            assert pred[(batch * bsize):(batch * bsize + bsize)].shape == y_pred.shape
            pred[(batch * bsize):(batch * bsize + bsize)] = y_pred # Fill slice

    return pred, loss_test


def eval_calib_tta(model: torch.nn.Module,
                   num_workers: int,
                   dataset_config: dict,
                   dataloader_config: dict,
                   model_str: str,
                   model_id: int,
                   parallel_use: bool,
                   data_raw_path: str,
                   device: str,
                   uq_method: str,
                   calibration_size: int = 500,
                   alpha: float = 0.1,
                   city_limit: Optional[int] = None,
                   to_file: bool = True,
                   **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name))

    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    post_transform = configs[model_str]["post_transform"][uq_method]
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
        
        pred, loss_city = evaluate(device=device,
                                   loss_fct=loss_fct,
                                   dataloader=dataloader,
                                   model=model,
                                   samp_limit=calibration_size,
                                   parallel_use=parallel_use,
                                   post_transform=post_transform,
                                   no_day_limit=True)
        logging.info(f"Obtained predictions as {pred.shape}.")
        logging.info(f"MSE loss for {city}: {loss_city}.")

        pred = aggregate_tta(pred) # (samples, 3, 6, H, W, Ch)
        logging.info(f"Aggregated predictions to obtain uncertainty via {uq_method} as {pred.shape}.")

        pred = get_quantile(pred, n=calibration_size, alpha=alpha).to(torch.float16) # (6, H, W, Ch)
        logging.info(f"Obtained quantiles as {pred.shape}.")

        if to_file:
            file = os.path.join(data_raw_path, city, f"calib_quant_{int((1-alpha)*100)}_tta_{model_str+str(model_id)}.h5")
            write_data_to_h5(data=pred, dtype=np.float16, filename=file, compression="lzf", verbose=True)
    logging.info(f"Written all calibration set {(1-alpha)*100}% quantiles for {cities=} to file.")

    del data, dataloader, pred


"""
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
                   uq_method: str,
                   dataset_limit: Optional[list] = None,
                   **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name

    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    post_transform = configs[model_str]["post_transform"][uq_method]
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
