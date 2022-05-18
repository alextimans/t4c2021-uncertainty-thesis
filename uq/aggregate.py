import os
from pathlib import Path

import numpy as np
import torch

from util.h5_util import load_h5_file, write_data_to_h5


def aggregate_tta_ens(pred_paths: dict, base_path: str = None, device: str = None):

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

        model_preds = [load_h5_file(paths[file_idx], dtype=torch.uint8) for paths in pred_paths.values()] # Load model preds into memory (big!)

        sample_counts = [pred.shape[0] for pred in model_preds]
        nr_samples = sample_counts[0]
        assert all(count == nr_samples for count in sample_counts) # Equal sample counts per model

        pred_uq = torch.empty(size=(nr_samples, 3, 6, 495, 436, 8), # 3: y_pred, epistemic, aleatoric
                              dtype=torch.float16, device=device) # Half-float precision for reduced memory consumption

        for samp_idx in range(nr_samples):
            samp_pred_uq = torch.empty(size=(3, 6, 495, 436, 8),
                                       dtype=torch.float16, device=device)
            samp_y_pred = torch.empty(size=(ensemble_size, 6, 495, 436, 8),
                                      dtype=torch.uint8, device=device)
            samp_aleatoric = torch.empty(size=(ensemble_size, 6, 495, 436, 8),
                                      dtype=torch.float16, device=device)

            for ens_idx in range(ensemble_size):
                model_pred = model_preds[ens_idx][samp_idx, ...] # [1+k, 6, 495, 436, 8] uint8
                samp_y_pred[ens_idx, ...] = model_pred[0, ...].unsqueeze(0) # pred for original img, [1, 6, 495, 436, 8] uint8

                # Aleatoric uncertainty estimation per model: std over original + augmented imgs
                samp_aleatoric[ens_idx, ...] = torch.std(model_pred.to(torch.float16), dim=0, unbiased=False, keepdim=True) # [1, 6, 495, 436, 8] float16

            del model_pred # Remove local context var from memory

            # Final point prediction: avg over ensemble preds for original img (use available ensemble for free performance improvement)
            samp_pred_uq[0, ...] = torch.mean(samp_y_pred.to(torch.float16), dim=0, keepdim=True) # [1, 6, 495, 436, 8] float16

            # Final epistemic uncertainty estimation: std over ensemble preds for original img
            samp_pred_uq[1, ...] = torch.std(samp_y_pred.to(torch.float16), dim=0, unbiased=False, keepdim=True) # [1, 6, 495, 436, 8] float16

            # Final aleatoric uncertainty estimation: mean over aleatoric uncertainties per model
            samp_pred_uq[2, ...] = torch.mean(samp_aleatoric, dim=0, keepdim=True) # [1, 6, 495, 436, 8] float16

            pred_uq[samp_idx, ...] = samp_pred_uq.unsqueeze(dim=0) # Add sample
            
        del samp_pred_uq, samp_y_pred, samp_aleatoric

        file_path = os.path.join(folder_path, file_name)
        write_data_to_h5(data=pred_uq, dtype=np.float16, filename=file_path, compression="lzf", verbose=True)

    del model_preds, pred_uq

"""Test

p = torch.ones(10, 8, 6, 495, 436, 8, dtype=torch.uint8)
# file_names should be the same in real scenario
write_data_to_h5(data=p, dtype=np.uint8, filename="unet_1_pred", compression="lzf")
write_data_to_h5(data=p, dtype=np.uint8, filename="unet_2_pred", compression="lzf")
write_data_to_h5(data=p, dtype=np.uint8, filename="unet_3_pred", compression="lzf")

pred_paths = {"unet_1": ["./data/test_pred_uq/unet_1/unet_1_pred.h5", "./data/test_pred_uq/unet_1/unet_1_pred2.h5"],
              "unet_2": ["./data/test_pred_uq/unet_2/unet_2_pred.h5", "./data/test_pred_uq/unet_2/unet_2_pred2.h5"],
              "unet_3": ["./data/test_pred_uq/unet_3/unet_3_pred.h5", "./data/test_pred_uq/unet_3/unet_3_pred2.h5"]}
"""
"""
calculate tensor memory size in byte: tensor.element_size() * tensor.nelement()

torch.zeros((288, 8, 6, 495, 436, 8)) has 95 GB ?! => as uint8: ~24GB
torch.zeros((288, 6, 495, 436, 8)) has ~12GB => as uint8: ~3GB
make sure to load preds as uint8 types, not float! dtype=torch.uint8
also instead of full float use half-float. dtype=torch.float16
    torch.bfloat16 has more limited support e.g. with h5 or numpy

use del variable to free up memory allocation once stg not needed anymore
https://pytorch.org/docs/stable/notes/faq.html
"""
"""
aggregation function aggregate_tta_ens

receives a dictionary of style
    {"unet_id": list of file_paths to pred files of shape [samples, 1+k, 6, 495, 436, 8]}
preds for each model need to be FOR THE SAME test data !
each key is interpreted as an independent ensemble model
each value is a list containing the augmented predictions for that model on
    a previously specified number of test files + samples
    e.g. for standard pipeline it is all files with 288 temporally ordered samples per file
    but for t4c paper it will be 2 files with 100 random samples per file

nr_files = len for first file_path
assert nr_files == len(list_file_paths) for each unet

for idx in len(nr_files): #FOR EACH FILE
    
    list_unet_files = [] # len = #unet_models
    for each unet_model:
        load the respectively indexed file into memory
        list_unet_files.append(load_h5(dict[unet_model][idx])) uint8 type
        
        infer #samples in file, assert that identical for each file

    tensor_final_pred = [samples, 3, 6, 495, 436, 8], float type -> 3: y_pred, epistemic, aleatoric
    
    for each sample_idx from indexed file range #FOR EACH SAMPLE IN FILE
        tensor_sample_pred = [3, 6, 495, 436, 8], float type #disregard first dim always 1
        
        tensor_samp_aleatoric = [#unet_models, 6, 495, 436, 8] float type
        tensor_samp_ypreds = [#unet_models, 6, 495, 436, 8] uint8

        for each unet model:
            - pred = file[sample_idx, ...]
            should result in [1+k, 6, 495, 436, 8] uint8 tensor

            - extract the pred for original img and add to tensor_samp_ypreds
            should result in [1, 6, 495, 436, 8] uint8 tensors being added
            - tensor_samp_ypreds[unet_idx, ...] = pred[0, ...]
            
            - compute variance over augmentations: torch.var(pred, dim_aug).unsqueeze(dim_aug)
            should result in [1, 6, 495, 436, 8] float tensor -> aleatoric for unet
            - add aleatoric to tensor_samp_aleatoric
        
        - compute mean over tensor_samp_ypreds -> y_pred [1, 6, 495, 436, 8] float tensor
            over #unet_models dimension
            make use of ensembling for improved point prediction since already available
        - add y_pred avg across unets to tensor_sample_pred

        - compute var over tensor_samp_ypreds -> epistemic [1, 6, 495, 436, 8] float tensor
            over #unet_models dimension
        - add epistemic to tensor_sample_pred

        - compute mean over aleatoric vars: torch.mean(tensor_samp_aleatoric, dim_models)
            over #unet_models dimension
        should result in [1, 6, 495, 436, 8] tensor -> aleatoric avg across unets
        - add aleatoric avg across unets to tensor_sample_pred
    
        - add tensor_sample_pred.unsqueeze(dim=0) to tensor_final_pred
        i.e. add [1, 3, 6, 495, 436, 8] to [samples, 3, 6, 495, 436, 8]
    
    - save tensor_final_pred as h5 file with same name as original file?
    went from [samples, 1+k, 6, 495, 436, 8] x 3 ==> [samples, 3, 6, 495, 436, 8]

"""
