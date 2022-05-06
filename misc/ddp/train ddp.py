# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import os
import logging
import sys
from typing import Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from model.early_stopping import EarlyStopping
from model.checkpointing import save_file_to_folder
from data.dataset import T4CDataset
from util.monitoring import system_status
from util.get_device import get_device

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


def run_model_ddp(rank: int,
                  model: torch.nn.Module,
                  data_train: T4CDataset,
                  data_val: T4CDataset,
                  dataloader_config: dict,
                  optimizer_config: dict,
                  lr_scheduler_config: dict,
                  earlystop_config: dict,
                  args: dict) -> Tuple[torch.nn.Module, str]:

    """
    memory pinning: https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data
                    https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers

    mp.spawn: https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses
    torch.distributed: hhttps://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    ddp ex: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    ddp ex: https://pytorch.org/docs/stable/notes/ddp.html
    ddp doc: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
    ddp blog: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    ddp blog: https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/#setting-up-the-same-model-with-distributeddataparallel

    euler parallel: https://scicomp.ethz.ch/wiki/Parallel_computing
                    https://scicomp.ethz.ch/wiki/Parallel_job_submission
    euler doc: "All GPUs are configured in Exclusive Process mode"
    cluster support: "Exclusive Process mode cannot be overriden by users"
                                         
    ERROR MSG: "RuntimeError: CUDA error: all CUDA-capable devices are busy or unavailable"
    ddp error: https://discuss.pytorch.org/t/distributeddataparallel-runtimeerror-cuda-error-all-cuda-capable-devices-are-busy-or-unavailable/102763
    ddp most likely error reason:
        https://forum.opennmt.net/t/all-cuda-capable-devices-are-busy-or-unavailable/3937
    possible solution (not possible on cluster): 
        https://stackoverflow.com/questions/31731535/switch-cuda-compute-mode-to-default-mode

    tried -R "span[ptile=5]" argument in euler but wasn't recognized properly, anyway I don't want to access mult. nodes
    
    tried cuda/11.0.3 => cuda/10.2.89 module load as similarly suggested here to no avail:
    https://discuss.pytorch.org/t/runtimeerror-cuda-error-all-cuda-capable-devices-are-busy-or-unavailable-with-distributeddataparallel/144589
    """

    random_seed = args["random_seed"]
    nr_gpus = args["nr_gpus"]
    batch_size = args["batch_size"]
    num_workers = args["num_workers"]
    epochs = args["epochs"]
    model_str = args["model_str"]
    model_id = args["model_id"]
    save_checkpoint = args["save_checkpoint"]
    data_parallel = args["data_parallel"]
    display_system_status = args["display_system_status"]
    device = args["device"]
#    device_ids = args["device_ids"]

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name

    dist.init_process_group("nccl", world_size=nr_gpus, rank=rank)

    # Load data
    train_sampler = DistributedSampler(dataset=data_train,
                                       num_replicas=nr_gpus,
                                       rank=rank,
                                       shuffle=True,
                                       seed=random_seed)
    val_sampler = DistributedSampler(dataset=data_val,
                                       num_replicas=nr_gpus,
                                       rank=rank,
                                       shuffle=True,
                                       seed=random_seed)

    train_loader = DataLoader(dataset=data_train,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              sampler=train_sampler,
                              pin_memory=True,
                              **dataloader_config)
    val_loader = DataLoader(dataset=data_val,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=val_sampler,
                            pin_memory=True,
                            **dataloader_config)
    logging.info(f"Created data loaders with {batch_size=}.")

    # Device setting + DDP model
    device, _ = get_device(device, data_parallel)
    device = torch.device(device, rank) # "cuda:rank"
    logging.info(f"Training on {device=}.")

    model = model.to(device, non_blocking=True)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    logging.info(f"Using GPU '{device}': {torch.cuda.get_device_name()}")

    # Loss function
    loss_fct = torch.nn.functional.mse_loss #torch.nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), **optimizer_config)
    # LR Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_config)
    # Early Stopping
    early_stopping = EarlyStopping(**earlystop_config)

    # Training
    loss_train, loss_val = train_model(device, epochs, optimizer, loss_fct,
                                       train_loader, val_loader, model, model_str,
                                       model_id, save_checkpoint, early_stopping,
                                       lr_scheduler, display_system_status)
    logging.info("Finished training of model %s on %s for %s epochs.",
                 model_str, device, epochs)
    logging.info("Final loss '{}' -> Train: {:.4f}, Val: {:.4f}"
                 .format(loss_fct.__name__, loss_train[-1], loss_val[-1]))

    #dist.destroy_process_group()

    return model, device

#TODO FIX MEMORY PIN AND NO BLOCKING TO ONLY WORK WHEN USING GPUS

def run_model(model: torch.nn.Module,
              data_train: T4CDataset,
              data_val: T4CDataset,
              batch_size: int,
              num_workers: int,
              epochs: int,
              dataloader_config: dict,
              optimizer_config: dict,
              lr_scheduler_config: dict,
              earlystop_config: dict,
              model_str: str,
              model_id: int,
              save_checkpoint: str,
              data_parallel: str,
              display_system_status: bool,
              device: str = None,
              device_ids = None,
              **kwargs) -> Tuple[torch.nn.Module, str]:

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name

    # Load data
    train_sampler = RandomSampler(data_train)
    train_loader = DataLoader(dataset=data_train,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              sampler=train_sampler,
                              pin_memory=True,
                              **dataloader_config)
    val_sampler = RandomSampler(data_val)
    val_loader = DataLoader(dataset=data_val,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=val_sampler,
                            pin_memory=True,
                            **dataloader_config)
    logging.info(f"Created data loaders with {batch_size=}.")

    # Device setting
    device, parallel_use = get_device(device, data_parallel)
    if parallel_use:
        # https://stackoverflow.com/questions/59249563/runtimeerror-module-must-have-its-parameters-and-buffers-on-device-cuda1-devi
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logging.info(f"Using {len(model.device_ids)} GPUs: {model.device_ids}.")
        device = f"cuda:{model.device_ids[0]}"

    logging.info(f"Training on {device=}.")
    model = model.to(device, non_blocking=True)

    # Loss function
    loss_fct = torch.nn.functional.mse_loss #torch.nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), **optimizer_config)
    # LR Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_config)
    # Early Stopping
    early_stopping = EarlyStopping(**earlystop_config)

    # Training
    loss_train, loss_val = train_model(device, epochs, optimizer, loss_fct,
                                       train_loader, val_loader, model, model_str,
                                       model_id, save_checkpoint, early_stopping,
                                       lr_scheduler, display_system_status)
    logging.info("Finished training of model %s on %s for %s epochs.",
                 model_str, device, epochs)
    logging.info("Final loss '{}' -> Train: {:.4f}, Val: {:.4f}"
                 .format(loss_fct.__name__, loss_train[-1], loss_val[-1]))

    return model, device


def train_model(device, epochs, optimizer, loss_fct, train_loader, val_loader,
                model, model_str, model_id, save_checkpoint, early_stopping,
                lr_scheduler, display_system_status) -> Tuple[list, list]:

    l_train, l_val = [], []
    for epoch in range(epochs):
        loss_train = _train_epoch(device, epoch, optimizer, loss_fct, train_loader, model)
        loss_val = _val_epoch(device, epoch, loss_fct, val_loader, model)
        l_train.append(loss_train); l_val.append(loss_val)

        logging.info("Epoch: {}, Train loss: {:.4f}, Val loss: {:.4f}"
                     .format(epoch, loss_train, loss_val))
        if eval(display_system_status) is not False:
            logging.info(system_status()) # Visualize GPU, memory, disk usage

        lr_scheduler.step(loss_val) # lr_scheduler.get_last_lr(), optimizer.param_groups[0]["lr"]
        early_stopping(model, loss_val, epoch, model_str, model_id, save_checkpoint)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at epoch {epoch}.")
            break

    folder_dir = os.path.join(save_checkpoint, f"{model_str}_{model_id}")
    save_file_to_folder(file=l_train, filename="loss_train", folder_dir=folder_dir,
                        fmt="%.4f", header=f"{model_str}_{model_id} trained until {epoch=}")
    save_file_to_folder(file=l_val, filename="loss_val", folder_dir=folder_dir,
                        fmt="%.4f", header=f"{model_str}_{model_id} trained until {epoch=}")

    return l_train, l_val


def _train_epoch(device, epoch, optimizer, loss_fct, dataloader, model) -> float:
    model.train()
    loss_sum = 0

    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred = model(X) # Shape: [batch_size, 6*8, 496, 448]
            loss = loss_fct(y_pred, y) # Mean over batch samples + channels + pixels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            loss_sum += float(loss.item())
            loss_train = float(loss_sum/(batch+1)) # Add mean over batches -> mean over all samples
            tepoch.set_description(f"Epoch {epoch} > train")
            tepoch.set_postfix(loss = loss_train)

    return loss_train


@torch.no_grad()
def _val_epoch(device, epoch, loss_fct, dataloader, model) -> float:
    model.eval()
    loss_sum = 0

    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred = model(X)
            loss = loss_fct(y_pred, y)

            loss_sum += float(loss.item())
            loss_val = float(loss_sum/(batch+1))
            tepoch.set_description(f"Epoch {epoch} > val")
            tepoch.set_postfix(loss = loss_val)

    return loss_val
