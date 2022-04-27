import logging
from typing import Tuple

import torch.cuda as cuda


def get_device(device: str = None, data_parallel: bool = False) -> Tuple[str, str]:

    logging.info(f"Requested {device=}.")
    gpu_available = cuda.is_available()
    parallel_use = False

    if (device is None) and (not gpu_available):
        logging.warning("Device not specified, 'cuda' not available, using 'cpu'.")
        device = "cpu"

    elif (device is None) and gpu_available:
        logging.warning("Device not specified, 'cuda' available and being used.")
        device = "cuda"

    elif (device=="cpu") and (not gpu_available): # Most common use case for local
        pass

    elif (device == "cpu") and gpu_available:
        logging.info("'cpu' used but 'cuda' is available.")

    elif (device == "cuda") and (not gpu_available):
        logging.warning("'cuda' requested but not available, using 'cpu' instead.")
        device = "cpu"

    elif (device == "cuda") and gpu_available:
        nr_devices = cuda.device_count()

        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        if data_parallel and (nr_devices > 1):
            logging.info(f"DataParallel on {nr_devices} GPUs possible.")
            parallel_use = True

        elif data_parallel and (not nr_devices > 1):
            logging.warning(f"{data_parallel=} but only {nr_devices} GPUs available.")

        elif (not data_parallel) and (nr_devices > 1):
            logging.info(f"{data_parallel=} but {nr_devices} GPUs available.")

        else: # Most common use case for cluster
            logging.info(f"Using {nr_devices} GPU: {cuda.get_device_name()}")

    else:
        pass

    return device, parallel_use

    """
    No device, no GPU -> CPU + warning
    No device, yes GPU -> GPU + warning
    CPU device, no GPU -> CPU
    CPU device, yes GPU -> CPU + info
    GPU device, no GPU -> CPU + warning
    GPU device, yes GPU
        parallel yes, GPU >1 -> info
        parallel yes, GPU =1 -> warning
        parallel no, GPU >1 -> info
        parallel no, GPU =1 -> info
    """
