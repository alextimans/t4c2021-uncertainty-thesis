import os
import sys
import logging
import glob
from pathlib import Path
import argparse

import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import combine_pvalues
from statsmodels.distributions.empirical_distribution import ECDF

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import T4CDataset
from model.configs import configs
from model.checkpointing import load_torch_model_from_checkpoint
from util.h5_util import write_data_to_h5, load_h5_file
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

    parser.add_argument("--out_bound", type=float, default=0.01, required=False,
                        help="Outlier decision boundary: if p-value <= out_bound we consider value an outlier.")
    parser.add_argument("--test_pred_bool", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if test_pred function should be called.")
    parser.add_argument("--detect_outliers_bool", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if detect_outliers function should be called.")

    return parser


def test_pred(model: torch.nn.Module,
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
    
        test_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/test/*8ch.h5", recursive=True))
        logging.info(f"{len(test_file_paths)} test files extracted from {data_raw_path}/{city}/test/...")

        if test_pred_path is None:
            raise AttributeError
        else:
            res_path = Path(os.path.join(test_pred_path, city))
            res_path.mkdir(exist_ok=True, parents=True)
        
        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=test_file_paths,
                          **dataset_config)

        # idx of fixed sample index for each file
        logging.info(f"Using fixed sample index {fix_samp_idx[i]} for {city}.")
        sub_idx = [fix_samp_idx[i]+t*288 for t in range(len(test_file_paths))]

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

        logging.info(f"Obtained test set preds with uncertainty {uq_method} as {pred.shape, pred.dtype}.")
        if pred_to_file:
            write_data_to_h5(data=pred, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(res_path, f"pred_{uq_method}.h5"))
        del data, dataloader, pred

        logging.info(f"Evaluation via {uq_method} finished for {city}.")
    logging.info(f"Evaluation via {uq_method} finished for all cities in {cities}.")


def detect_outliers(model: torch.nn.Module,
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
                out_bound: float,
                pred_to_file: bool = True,
                pval_to_file: bool = True,
                out_to_file: bool = True,
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
            raise AttributeError
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

        pred_tr, loss_city = uq_method_obj(device=device, # (samples, 3, H, W, Ch) torch.float32
                                       loss_fct=loss_fct,
                                       dataloader=dataloader,
                                       model=model,
                                       samp_limit=len(sub_idx),
                                       parallel_use=parallel_use,
                                       post_transform=post_transform)

        logging.info(f"Obtained train set preds with uncertainty {uq_method} as {pred_tr.shape, pred_tr.dtype}.")
        if pred_to_file:
            write_data_to_h5(data=pred_tr, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(res_path, f"pred_tr_{uq_method}.h5"))
        del data, dataloader

        unc_tr = pred_tr[:, 2, ...].to("cpu") # Train set uncertainties
        del pred_tr
        pred_path = os.path.join(test_pred_path, city, f"pred_{uq_method}.h5")
        unc = load_h5_file(pred_path, dtype=torch.float32)[:, 2, ...].to("cpu") # Test set uncertainties

        # Cell-level uncertainty KDE Gaussian fit + p-values
        pval = get_pvalues(unc_tr=unc_tr, unc=unc, device=device)
        logging.info(f"Obtained tensor of p-values as {pval.shape, pval.dtype}.")
        if pval_to_file:
            write_data_to_h5(data=pval, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(res_path, f"pval_{uq_method}.h5"))
        del unc_tr, unc

        # p-value aggregation and outlier labelling (channel-level, pixel-level)
        out = aggregate_pvalues(pval, out_bound, device)
        logging.info(f"Obtained tensor of outlier labels as {out.shape, out.dtype}.")
        if out_to_file:
            write_data_to_h5(data=out, dtype=bool, compression="lzf", verbose=True,
                             filename=os.path.join(res_path, f"out_{uq_method}.h5"))

        # Outlier detection stats
        outlier_stats(out)
        del pval, out

        logging.info(f"Outlier detection via {uq_method} finished for {city}.")
    logging.info(f"Outlier detection via {uq_method} finished for all cities in {cities}.")


def get_pvalues(unc_tr, unc, device: str):
    samp, p_i, p_j, channels = tuple(unc.shape)
    # Tensor containing cell-level p-values
    # for test set uncertainty vs. train set uncertainty KDE fit
    pval = torch.empty(size=(samp, p_i, p_j, channels), dtype=torch.float32, device="cpu")

    for i in tqdm(range(p_i), desc="Pixel height"):
        for j in range(p_j):
            for ch in range(channels):

                cell_tr = unc_tr[:, i, j, ch]
                cell = unc[:, i, j, ch]

                kde = gaussian_kde(cell_tr, bw_method="scott")
                sfit = kde.resample(size=100000).reshape(-1)
                med = np.median(sfit)

                # Empirical CDF of KDE fit from large sample for support set coverage
                ecdf = ECDF(sfit)
                p = ecdf(cell) # array of CDF prob values across sample dim
                p[cell > med] = 1 - p[cell > med] # array of p-values
                pval[:, i, j, ch] = torch.tensor(p, dtype=torch.float32)

                del cell_tr, cell, kde, sfit, ecdf

    assert pval.max() <= 1 and pval.min() >= 0, "p-values not in [0, 1]"
    return pval


def aggregate_pvalues(pval, out_bound: float, device: str):
    logging.info(f"Using outlier decision boundary {out_bound=}.")
    samp, p_i, p_j, _ = tuple(pval.shape)
    # Boolean tensor containing outlier labelling
    out = torch.empty(size=(samp, p_i, p_j, 3), dtype=torch.bool, device="cpu")

    for i in tqdm(range(p_i), desc="Pixel height"):
        for j in range(p_j):
            for s in range(samp):

                out_ch = aggregate_channels(pval[s, i, j, :], out_bound)
                out_pix = aggregate_pixel(out_ch)
                # 1: Outlier vol, 2: Outlier speed, 3: Outlier pixel
                out[s, i, j, :] = torch.cat((out_ch, out_pix))
    return out


def aggregate_channels(pval, out_bound: float):
    # Channel group is outlier if combined p-value outside outlier bound
    p_vol_agg = combine_pvalues(pval[[0, 2, 4, 6]], method="fisher")
    p_sp_agg = combine_pvalues(pval[[1, 3, 5, 7]], method="fisher")

    out_vol = True if p_vol_agg <= out_bound else False
    out_sp = True if p_sp_agg <= out_bound else False

    return torch.tensor([out_vol, out_sp])


def aggregate_pixel(out_ch):
    # Pixel is outlier if at least one channel group is outlier
    out_pix = True if out_ch.sum() > 0 else False

    return torch.tensor([out_pix])


def outlier_stats(out):
    samp, p_i, p_j, _ = tuple(out.shape)

    logging.info("### Outlier stats ###")
    logging.info(f"Outliers by vol ch: {out[..., 0].sum()}")
    logging.info(f"Outliers by speed ch: {out[..., 1].sum()}")
    logging.info(f"Outliers by pixel: {out[..., 2].sum()}/{p_i*p_j}")

    i, j = (out[..., 2].sum(dim=0) == out[..., 2].sum(dim=0).max()).nonzero().squeeze()
    v = out[..., 2].sum(dim=0)[i, j]
    logging.info(f"Pixel with most outliers by sample: {(i.item(), j.item())} with {v}/{samp} outliers.")

    samp_v = out[..., 2].sum(dim=(1,2))
    logging.info(f"Mean pixel outlier count across samples: {samp_v.mean()}.")
    logging.info(f"Sample with most pixel outliers: test sample {samp_v.argmax().item()+1} with {samp_v[samp_v.argmax()].item()} outliers.")
    logging.info(f"Sample with least pixel outliers: test sample {samp_v.argmin().item()+1} with {samp_v[samp_v.argmin()].item()} outliers.")


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
    test_pred_bool = args.test_pred_bool
    detect_outliers_bool = args.detect_outliers_bool

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

    if eval(test_pred_bool) is not False:
        logging.info("Collecting test set preds, in particular uncertainties.")
        test_pred(model=model,
                       cities=["BANGKOK", "BARCELONA", "MOSCOW"],
                       dataset_config=dataset_config,
                       dataloader_config=dataloader_config,
                       parallel_use=parallel_use,
                       device=device,
                       **(vars(args)))
        logging.info("Test set preds collected.")
    else:
        logging.info("Test set preds assumed to be available.")

    if eval(detect_outliers_bool) is not False:
        logging.info("Detecting outliers on test set via train set distr. fits.")
        detect_outliers(model=model,
                       cities=["BANGKOK", "BARCELONA", "MOSCOW"],
                       dataset_config=dataset_config,
                       dataloader_config=dataloader_config,
                       parallel_use=parallel_use,
                       device=device,
                       **(vars(args)))
        logging.info("Outliers detected.")
    else:
        logging.info("No outlier detection occuring.")
    logging.info("Main finished.")


if __name__ == "__main__":
    main()
