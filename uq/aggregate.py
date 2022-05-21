import os
from pathlib import Path

import numpy as np
import torch

from util.h5_util import load_h5_file, write_data_to_h5


def aggregate_tta(pred):

    """
    Receives: prediction tensor (samples, 1+1+k, 6, H, W, Ch),
    computes the aleatoric uncertainty obtained via test-time augmentation.
    Returns: tensor (samples, 3, 6, H, W, Ch) where 2nd dimension
    '3' is y_true (0), point prediction (1), aleatoric uncertainty (2).
    """

    # Aleatoric uncertainty estimation: std over original + augmented imgs
    pred[:, 2, ...] = torch.std(pred[:, 1:, ...], dim=1, unbiased=False)

    return pred[:, :3, ...]


############ old code (to be revisited)

def aggregate_tta_ens2(pred_paths: dict, base_path: str = None, device: str = None):

    """
    Aggregation function that receives prediction files and computes the
    point estimate, aleatoric uncertainty via test-time data augmentation
    and epistemic uncertainty via deep ensembling.

    Receives prediction files of format [samples, 1+1+k, 6, 495, 436, 8] x ensemble_size
    and saves file with respective quantities of form [samples, 4, 6, 495, 436, 8],
    where 2nd dimension '4' indicates y_true (0), point estimate (1), aleatoric (2), epistemic (3).
    Prediction files should be for an individual city at a time.
    1+1+k = y_true + pred original img + pred augmented imgs

    Parameters
    ----------
    pred_paths: dict
        Dictionary of format {"unet_id": [pred_file_paths]}, where pred_file_paths are
        paths that lead to pred files for the given model of shape [samples, 1+k, 6, 495, 436, 8].
        Each ensemble member has a respective entry in the dictionary.
    base_path: str
        Base path used to store the uncertainty results in. 
        Recommended same base folder as the predictions are located in.
    device: str
        Device to load tensors onto.
    """

    if base_path is None:
        base_path = "./data/test_pred_uq"
    folder_path = os.path.join(base_path, "unet_tta_ens")
    Path(folder_path).mkdir(exist_ok=True, parents=True)

    ensemble_size = len(pred_paths.keys())

    file_counts = [len(paths) for paths in list(pred_paths.values())]
    nr_files = file_counts[0]
    assert all(count == nr_files for count in file_counts) # Equal file counts per model

    for file_idx in range(nr_files):

        file_names = [os.path.split(paths[file_idx])[-1] for paths in pred_paths.values()]
        file_name = file_names[0]
        assert all(names == file_name for names in file_names) # Equal file name for each model

        model_preds = [load_h5_file(paths[file_idx], dtype=torch.uint8) for paths in pred_paths.values()] # Load model preds into memory (very big!) list of [samples, 1+k, 6, 495, 436, 8] uint8

        sample_counts = [pred.shape[0] for pred in model_preds]
        nr_samples = sample_counts[0]
        assert all(count == nr_samples for count in sample_counts) # Equal sample counts per model

        pred_uq = torch.empty(size=(nr_samples, 3, 6, 495, 436, 8), # 4: y_true, y_pred, aleatoric, epistemic
                              dtype=torch.float16, device=device) # Half-float precision for reduced memory consumption

        for samp_idx in range(nr_samples):
            samp_y_pred = torch.empty(size=(ensemble_size, 6, 495, 436, 8),
                                      dtype=torch.uint8, device=device)
            samp_aleatoric = torch.empty(size=(ensemble_size, 6, 495, 436, 8),
                                      dtype=torch.float16, device=device)

            for ens_idx in range(ensemble_size):
                samp_y_pred[ens_idx, ...] = model_preds[ens_idx][samp_idx, 0, ...] # pred for original img [6, 495, 436, 8] uint8

                # Aleatoric uncertainty estimation per model: std over original + augmented imgs
                samp_aleatoric[ens_idx, ...] = torch.std(model_preds[ens_idx][samp_idx, ...].to(torch.float16), dim=0, unbiased=False) # [6, 495, 436, 8] float16

            # Final point prediction: avg over ensemble preds for original img (use available ensemble for free performance improvement)
            pred_uq[samp_idx, 0, ...] = torch.mean(samp_y_pred.to(torch.float16), dim=0) # [6, 495, 436, 8] float16

            # Final aleatoric uncertainty estimation: mean over aleatoric uncertainties per model
            pred_uq[samp_idx, 1, ...] = torch.mean(samp_aleatoric, dim=0) # [6, 495, 436, 8] float16

            # Final epistemic uncertainty estimation: std over ensemble preds for original img
            pred_uq[samp_idx, 2, ...] = torch.std(samp_y_pred.to(torch.float16), dim=0, unbiased=False) # [6, 495, 436, 8] float16

        del samp_y_pred, samp_aleatoric

        file_path = os.path.join(folder_path, file_name)
        write_data_to_h5(data=pred_uq, dtype=np.float16, filename=file_path, compression="lzf", verbose=True)

    del model_preds, pred_uq


def aggregate_tta2(pred_paths: dict, base_path: str = None, device: str = None):

    """
    Aggregation function that receives prediction files and computes the
    point estimate and aleatoric uncertainty via test-time data augmentation.

    Receives prediction files of format [samples, 1+1+k, 6, 495, 436, 8] 
    and saves file with respective quantities of form [samples, 3, 6, 495, 436, 8],
    where 2nd dimension '3' indicates y_true (0), point estimate (1), aleatoric (2).
    1+1+k = y_true + pred original img + pred augmented imgs.

    Parameters
    ----------
    pred_paths: dict
        Dictionary of format {"unet_id": [pred_file_paths]}, where pred_file_paths are
        paths that lead to pred files for the given model of shape [samples, 1+1+k, 6, 495, 436, 8].
        Only a single entry in the dictionary for the given model.
    base_path: str
        Base path used to store the uncertainty results in. 
        Recommended same base folder as the predictions are located in.
    device: str
        Device to load tensors onto.
    """

    assert len(pred_paths.keys()) == 1 and len(pred_paths.items()) == 1
    ((model_str, file_paths),) = pred_paths.items()

    if base_path is None:
        base_path = "./data/test_pred_uq"
    folder_path = os.path.join(base_path, model_str + "_tta")
    Path(folder_path).mkdir(exist_ok=True, parents=True)

    for file_idx in range(len(file_paths)):

        file_name = os.path.split(file_paths[file_idx])[-1]

        model_pred = load_h5_file(file_paths[file_idx], dtype=torch.uint8) # Load model pred into memory (big!) [samples, 1+1+k, 6, 495, 436, 8] uint8
        nr_samples = model_pred.shape[0]

        pred_uq = torch.empty(size=(nr_samples, 3, 6, 495, 436, 8), # 3: y_true, y_pred, aleatoric
                              dtype=torch.float16, device=device) # Half-float precision for reduced memory consumption

        for samp_idx in range(nr_samples):

            # y_true
            pred_uq[samp_idx, 0, ...] = model_pred[samp_idx, 0, ...] # [6, 495, 436, 8] float16

            # Point prediction: prediction for original img
            pred_uq[samp_idx, 1, ...] = model_pred[samp_idx, 1, ...] # [6, 495, 436, 8] float16

            # Aleatoric uncertainty estimation: std over original + augmented imgs
            pred_uq[samp_idx, 2, ...] = torch.std(model_pred[samp_idx, 1:, ...].to(torch.float16), dim=0, unbiased=False) # [6, 495, 436, 8] float16

        file_path = os.path.join(folder_path, file_name)
        write_data_to_h5(data=pred_uq, dtype=np.float16, filename=file_path, compression="lzf", verbose=True)

    del model_pred, pred_uq


"""
p = torch.ones(10, 8, 6, 495, 436, 8, dtype=torch.uint8)
# file_names should be the same in real scenario
write_data_to_h5(data=p, dtype=np.uint8, filename="unet_1_pred", compression="lzf")
write_data_to_h5(data=p, dtype=np.uint8, filename="unet_2_pred", compression="lzf")
write_data_to_h5(data=p, dtype=np.uint8, filename="unet_3_pred", compression="lzf")

pred_paths = {"unet_1": ["./data/test_pred_uq/unet_1/unet_1_pred.h5", "./data/test_pred_uq/unet_1/unet_1_pred2.h5"],
              "unet_2": ["./data/test_pred_uq/unet_2/unet_2_pred.h5", "./data/test_pred_uq/unet_2/unet_2_pred2.h5"],
              "unet_3": ["./data/test_pred_uq/unet_3/unet_3_pred.h5", "./data/test_pred_uq/unet_3/unet_3_pred2.h5"]}

c = torch.Tensor([[[1,1],[1,1],[1,1]],[[0,0],[0,0],[1,1]],[[1,1],[2,2],[3,3]]])
"""
"""
calculate tensor memory size in byte: tensor.element_size() * tensor.nelement()
torch.zeros((288, 8, 6, 495, 436, 8)) has 95 GB ?! => as uint8: ~24GB
torch.zeros((288, 6, 495, 436, 8)) has ~12GB => as uint8: ~3GB
"""
