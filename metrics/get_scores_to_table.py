import os
# import glob
import argparse

import numpy as np
import pandas as pd

from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY
from metrics.get_scores import get_score_names


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_pred_path", type=str, default=None, required=True,
                        help="Test pred path.")

    return parser


def get_scores_to_table(test_pred_path: str):

    scorenames = get_score_names()
    colnames = list(np.delete(scorenames[1:-1].split(", "), [4,5,6,7]))
    colnames.insert(0, "uq_method")
    colnames.insert(0, "city")
    
    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]
    # limit to actually available cities as determined by folder structure
    # cities = cities[:len(glob.glob(f"{test_pred_path}/scores/*", recursive=True))]
    
    mask = ["", "_mask"]
    channels = ["speed", "vol"]
    uq_methods = ["point", "ensemble", "bnorm", "tta", "patches"]
    
    for m in mask:
        for ch in channels:
    
            df_list_of_lists = []
    
            for city in cities:
                for uq in uq_methods:
    
                    filename = f"scores_{uq}_{ch}{m}.txt"
                    try:
                        scores = list(np.loadtxt(os.path.join(test_pred_path, "scores", city, filename)))
                    except OSError:
                        scores = ["N/A" for i in range(13)]

                    scores_df = []
                    scores_df.append(city.lower())
                    scores_df.append(uq)
                    scores_df.append(str(scores[0]) + " +/- " + str(scores[4]))
                    scores_df.append(str(scores[1]) + " +/- " + str(scores[5]))
                    scores_df.append(str(scores[2]) + " +/- " + str(scores[6]))
                    scores_df.append(str(scores[3]) + " +/- " + str(scores[7]))
                    scores_df.append(str(scores[8]))
                    scores_df.append(str(scores[9]))
                    scores_df.append(str(scores[10]))
                    scores_df.append(str(scores[11]))
                    scores_df.append(str(scores[12]))
    
                    df_list_of_lists.append(scores_df)

            df = pd.DataFrame(df_list_of_lists, columns=colnames)
            dfname = f"results_{ch}{m}.csv"
            df.to_csv(os.path.join(test_pred_path, "scores", dfname), index=False, na_rep='N/A')
    
    # path = sorted(glob.glob(f"{test_pred_path}/scores/{city}/scores_{uq_method}_*.txt", recursive=True))
    # s = path[0]
    # s.split("/")[-1][:-4].split("_")


def main():
    parser = create_parser()
    args = parser.parse_args()
    get_scores_to_table(args.test_pred_path)


if __name__ == "__main__":
    main()
