import os
import glob
import argparse
from pathlib import Path

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm

from util.h5_util import load_h5_file


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--fig_path", type=str, default="./figures", required=False,
                        help="Base directory to store plots in.")
    parser.add_argument("--test_pred_path", type=str, default=None, required=False,
                        help="Directory from which to extract pred files.")
    parser.add_argument("--uq_method", type=str, default=None, required=False, choices=["ensemble", "bnorm"],
                        help="Specify UQ method for epistemic uncertainty.")
    parser.add_argument("--pixel", nargs=2, type=int, default=[None, None], required=False,
                        help="Pixel height and width (H W) for which to plot.")
    parser.add_argument("--mask_nonzero", type=str, default="False", required=False, choices=["True", "False"],
                        help="'Boolean' to mask for nonzero values.")

    return parser


def epistemic_hist(fig_path: str,
                   test_pred_path: str,
                   uq_method: str,
                   pixel: list,
                   cities: list,
                   mask_nonzero: str):

    assert pixel[0] <= 495 and pixel[1] <= 436, f"pixel value {pixel} not valid"
    mask = eval(mask_nonzero)

    for city in cities:

        path = os.path.join(test_pred_path, city, f"pred_{uq_method}.h5")
        pred = load_h5_file(path)

        # uncertainty for given pixel
        unc_vol = pred[:, 1, pixel[0], pixel[1], [0, 2, 4, 6]]
        unc_speed = pred[:, 1, pixel[0], pixel[1], [1, 3, 5, 7]]

        fig_dir = Path(os.path.join(fig_path, city))
        fig_dir.mkdir(exist_ok=True, parents=True)

        fig, (axs1, axs2) = plt.subplots(2, 4, sharey="row")

        for i in range(4):
            ts_vol = unc_vol[:, i].numpy()
            ts_speed = unc_speed[:, i].numpy()

            if mask:
                ts_vol = ts_vol[ts_vol > 0]
                ts_speed = ts_speed[ts_speed > 0]

            axs1[i].hist(ts_vol, bins=15, alpha=0.6, color="blue", density=True)

            mu, std = np.mean(ts_vol), np.std(ts_vol)
            #ts_min, ts_max = ts_vol.min(), ts_vol.max()
            x = np.linspace(mu-3*std, mu+3*std, 100)
            axs1[i].plot(x, norm.pdf(x, mu, std), color="red",
                         label="N({:.2f},{:.2f}".format(mu, std))

            axs1[i].legend(loc="upper right", fontsize="small")

            axs2[i].hist(ts_speed, bins=15, alpha=0.6, color="blue", density=True)

            mu, std = np.mean(ts_speed), np.std(ts_speed)
            #ts_min, ts_max = ts_speed.min(), ts_speed.max()
            x = np.linspace(mu-3*std, mu+3*std, 100)
            axs2[i].plot(x, norm.pdf(x, mu, std), color="red",
                         label="N({:.2f},{:.2f}".format(mu, std))

            axs2[i].legend(loc="upper right", fontsize="small")

        fig.suptitle(f"UQ: {uq_method}, {city}, pixel {pixel},\nmask>0:{mask}, Vol (row 1), Speed (row 2)", fontsize="small")

        f = f"{uq_method}_H{pixel[0]}_W{pixel[1]}"
        if mask:
            f += "_masked"
        fig_p = os.path.join(fig_dir, f)
        plt.savefig(fig_p, dpi=300, format="png", bbox_inches='tight')
        print(f"Saved figure for {city} at {fig_p}.")


def main():
    parser = create_parser()
    args = parser.parse_args()
    epistemic_hist(
        cities=["BANGKOK", "BARCELONA", "MOSCOW"],
        **(vars(args)))
    print("Main finished.")


if __name__ == "__main__":
    main()
