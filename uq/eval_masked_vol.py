import os
import sys
import logging
import argparse

import numpy as np

from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY
from model.checkpointing import save_file_to_folder
from util.h5_util import write_data_to_h5, load_h5_file

from metrics.get_scores import get_scores, get_score_names, get_scalar_scores


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    parser.add_argument("--cities", nargs="+", type=str, default=[None, None, None, None], required=False,
#                        help="Limit cities for which to generate test index files.")
    parser.add_argument("--data_raw_path", type=str, default="./data/raw", required=False,
                        help="Base directory of raw data.")
    parser.add_argument("--device", type=str, default="cpu", required=False, choices=["cpu", "cuda"],
                        help="Specify usage of specific device.")
    return parser


def eval_masked_vol(data_raw_path: str,
                    scores_to_file: bool = True,
                    device: str = "cpu", **kwargs):

    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]
    uq_methods = ["bnorm", "ensemble", "tta", "point", "patches"]
    folders = ["bnorm_unet2", "ensemble_unet2", "tta_unet2", "point_unet2", "patches_unet_patches6"]

    for city in cities:
        for i, uq_method in enumerate(uq_methods):

            logging.info(f"Evaluating UQ methods in {uq_methods} for {city}...")

            folder_path = os.path.join(data_raw_path, city, f"test_{folders[i]}")
            #folder_path = os.path.join("results", "calib_100_test_100_masked", city)
            pred = load_h5_file(os.path.join(folder_path, f"pred_{uq_method}.h5"))
            pred_interval = load_h5_file(os.path.join(folder_path, f"pi_{uq_method}.h5"))

            # which pixels have sum of vol > 0 across sample dimension
            mask_vol = pred[:, 0, :, :, [0, 2, 4, 6]].sum(dim=(0, -1)) > 0

            scores = get_scores(pred[..., mask_vol, :].unsqueeze(dim=-2),
                                pred_interval[..., mask_vol, :].unsqueeze(dim=-2)
                                )
            if scores_to_file:
                write_data_to_h5(data=scores, dtype=np.float16, compression="lzf", verbose=True,
                                 filename=os.path.join(folder_path, f"scores_{uq_method}_masked.h5"))
            score_names = get_score_names()
            scalar_speed, scalar_vol = get_scalar_scores(scores, device)
            del scores
            
            logging.info(f"Scores ==> {score_names}")
            logging.info(f"Scores for pred horizon 1h and speed channels masked: {scalar_speed}")
            logging.info(f"Scores for pred horizon 1h and volume channels masked: {scalar_vol}")
            save_file_to_folder(file=scalar_speed.cpu().numpy(), filename=f"scalar_scores_speed_{uq_method}_masked",
                                folder_dir=folder_path, fmt="%.4f", header=score_names)
            save_file_to_folder(file=scalar_vol.cpu().numpy(), filename=f"scalar_scores_vol_{uq_method}_masked",
                                folder_dir=folder_path, fmt="%.4f", header=score_names)

            logging.info(f"Evaluation via {uq_method} finished for {city}.")
    logging.info(f"Evaluation of UQ methods finished for all cities in {cities}.")


def main():
    parser = create_parser()
    args = parser.parse_args()
    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    eval_masked_vol(#cities=args.cities,
                    data_raw_path=args.data_raw_path,
                    device=args.device)


if __name__ == "__main__":
    main()
