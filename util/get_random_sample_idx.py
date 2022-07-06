import argparse
import random
import os
import numpy as np
from pathlib import Path
from typing import Union
from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cities", nargs="+", type=str, default=[None, None, None, None], required=False,
                        help="Limit cities for which to generate test index files.")
    parser.add_argument("--samp_size", type=int, default=100, required=False,
                        help="#samples for test run.")
    return parser


def get_random_sample_idx(citylist: list, samp_size: int):

    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]
    if not any(citylist):
        citylist = cities
    cities = list(set(cities).intersection(set(citylist)))

    for city in cities:
        if city == "ANTWERP":
            sub_idx = random.sample(range(25920, 51840), samp_size) # antwerp 2020
        else:
            sub_idx = random.sample(range(0, 25920), samp_size) # other cities 2020

        save_file_to_folder(sub_idx, "test_indices", folder_dir=f"./data/raw/{city}")


def save_file_to_folder(file = None, filename: str = None,
                        folder_dir: Union[Path, str] = None, **kwargs):

    folder_path = Path(folder_dir) if isinstance(folder_dir, str) else folder_dir
    folder_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(os.path.join(folder_path, f"{filename}.txt"), file, **kwargs)


def main():
    parser = create_parser()
    args = parser.parse_args()
    get_random_sample_idx(citylist=args.cities, samp_size=args.samp_size)


if __name__ == "__main__":
    main()
