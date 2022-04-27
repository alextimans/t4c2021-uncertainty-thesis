import os
import sys
import logging
import datetime
import glob
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

#from metrics.mse import mse
from data.dataset import T4CDataset
from util.h5_util import write_data_to_h5
from model.configs import configs
from model.checkpointing import save_file_to_folder
from data.data_layout import CITY_NAMES, MAX_FILE_DAY_IDX, TWO_HOURS


def eval_model(model: torch.nn.Module,
               batch_size: int,
               num_workers: int,
               dataset_config: dict,
               dataloader_config: dict,
               model_str: str,
               model_id: int,
               save_checkpoint: str,
               data_raw_path: str,
               test_pred_path: str,
               device: str = None,
               dataset_limit: Optional[list] = None,
               **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name

    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    post_transform = configs[model_str].get("post_transform", None)
    cities = CITY_NAMES
    loss_sum = []

    if dataset_limit[0] is not None: # Limit #cities
        assert dataset_limit[0] <= len(cities)
        cities = cities[:dataset_limit[0]]

    logging.info(f"Evaluating '{model_str}' on '{device}' with '{loss_fct.__name__}' for {cities}.")

    for city in cities:
    
        test_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/test/*8ch.h5", recursive=True))
        logging.info(f"Test files extracted from {data_raw_path}/{city}/test/...")

        if test_pred_path is None:
            city_pred_path = Path(f"{data_raw_path}/{city}/test_pred")
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
        logging.info(f"Saving predictions on the {nr_files} " +
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
#                    logging.info(f"{data.__len__()=}")

                    sampler = SequentialSampler(data)
                    dataloader = DataLoader(dataset=data,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            sampler=sampler,
                                            **dataloader_config)

                    pred, loss_file = evaluate(device=device,
                                               loss_fct=loss_fct,
                                               dataloader=dataloader,
                                               model=model,
                                               samp_limit=samp_limit)
                    loss_city.append(loss_file)

                    if post_transform is not None:
                        pred = post_transform(pred)

                    pred = pred.detach().numpy() # pred.cpu().detach().numpy()
#                    logging.info(f"{pred.shape=}")
                    temp_h5 = os.path.join(tmpdir, f"pred_{city}_samp{idx}")

                    write_data_to_h5(data=pred, filename=temp_h5)
                    logging.info(f"Pred for file {idx+1}/{nr_files} written to .h5.")

                    arcname = str(file_filter[0]).split("/")[-1] # e.g. '2019-06-04_ANTWERP_8ch.h5'
                    zipf.write(filename=temp_h5, arcname=arcname)
#                    logging.info(f"Added file as {arcname} to .zip.")

        logging.info(f"Written all {nr_files} pred files for {city} to .zip.")
#       logging.info(zipf.namelist())
        zipf_mb_size = os.path.getsize(zip_file_path) / (1024 * 1024)
        logging.info(f"Zip file '{zip_file_path}' of size {zipf_mb_size:.1f} MB.")

        loss_sum.append(loss_city) #loss_sum += (loss_city/nr_files)
        logging.info("Loss for {}: {:.4f}".format(city, np.mean(loss_city)))

    logging.info("Loss over all cities: {:.4f}".format(np.mean(loss_sum)))
    folder_dir = os.path.join(save_checkpoint, f"{model_str}_{model_id}")
    comment = f"rows: loss per city, cols: loss per file for that city, {dataset_limit=}"
    save_file_to_folder(file=loss_sum, filename="loss_test", folder_dir=folder_dir,
                        fmt="%.4f", header=comment)


@torch.no_grad()
def evaluate(device, loss_fct, dataloader, model, samp_limit) -> Tuple[torch.Tensor, float]:
    model.eval()
    loss_sum = 0

    bsize = dataloader.batch_size
    batch_limit = min(MAX_FILE_DAY_IDX, samp_limit) // bsize # Only predict for batches up to idx 288

    ds = dataloader.dataset.__getitem__(0)[1].size() # torch.Size([48, 496, 448])
    pred = torch.empty(size=(batch_limit * bsize, ds[0], ds[1], ds[2]),
                       dtype=torch.float, device=device)

    with tqdm(dataloader) as tloader:
        for batch, (X, y) in enumerate(tloader):
            if batch == batch_limit:
                break

            X, y = X.to(device), y.to(device)
            y_pred = model(X) # Shape: [batch_size, 6*8, 496, 448]
            loss = loss_fct(y_pred, y)

            loss_sum += float(loss.item())
            loss_test = float(loss_sum/(batch+1))
            tloader.set_description(f"Batch {batch+1}/{batch_limit} > eval")
            tloader.set_postfix(loss = loss_test)

            # Throws error for last batch if batch_size % 2 != 0
            assert pred[(batch * bsize):(batch * bsize + bsize)].shape == y_pred.shape
            pred[(batch * bsize):(batch * bsize + bsize)] = y_pred # Fill slice with batch preds

    return pred, loss_test
