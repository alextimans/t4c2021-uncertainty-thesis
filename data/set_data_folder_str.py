import os
import shutil
import glob
import logging
import argparse

from util.logging import t4c_apply_basic_logging_config
from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY, CITY_TRAIN_VAL_TEST, CITY_VAL_TEST_ONLY


def set_data_folder_str(base_path: str = None):

    for city in CITY_NAMES:
    
        if city in CITY_TRAIN_ONLY:
            shutil.move(os.path.join(base_path, city, "training"),
                        os.path.join(base_path, city, "train"))
    
        elif city in CITY_TRAIN_VAL_TEST:
            #paths_train = sorted(glob.glob(f"{base_path}/{city}/training/2019*8ch.h5"))
            paths_val_test = sorted(glob.glob(f"{base_path}/{city}/training/2020*8ch.h5"))
    
            split = len(paths_val_test) // 2
            paths_val, paths_test = paths_val_test[:split], paths_val_test[split:]
            assert len(paths_val) == len(paths_test)
    
            #train_folder = os.path.join(base_path, city, "train")
            val_folder = os.path.join(base_path, city, "val")
            test_folder = os.path.join(base_path, city, "test")
            #os.makedirs(train_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)
    
            for idx in range(split):
                shutil.move(paths_val[idx], val_folder)
                shutil.move(paths_test[idx], test_folder)
    
            shutil.move(os.path.join(base_path, city, "training"),
                        os.path.join(base_path, city, "train"))
    
        elif city in CITY_VAL_TEST_ONLY:
            paths_2019 = sorted(glob.glob(f"{base_path}/{city}/training/2019*8ch.h5"))
            paths_2020 = sorted(glob.glob(f"{base_path}/{city}/training/2020*8ch.h5"))
            assert len(paths_2019) == len(paths_2020)
    
            split = len(paths_2019) // 2
            paths_val_2019, paths_test_2019 = paths_2019[:split], paths_2019[split:]
            paths_val_2020, paths_test_2020 = paths_2020[:split], paths_2020[split:]
    
            val_folder = os.path.join(base_path, city, "val")
            test_folder = os.path.join(base_path, city, "test")
    
            os.makedirs(val_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)
    
            for idx in range(split):
                shutil.move(paths_val_2019[idx], val_folder)
                shutil.move(paths_val_2020[idx], val_folder)
                shutil.move(paths_test_2019[idx], test_folder)
                shutil.move(paths_test_2020[idx], test_folder)
    
            shutil.rmtree(os.path.join(base_path, city, "training"))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI args to set data folder structure.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_raw_path", type=str, default="./data/raw", required=False,
                        help="Base directory of raw data.")

    return parser


def main():
    t4c_apply_basic_logging_config()
    parser = create_parser()
    args = parser.parse_args()
    logging.info("Rearranging data into train / val / test folders...")
    set_data_folder_str(args.data_raw_path)
    logging.info("Data folders created.")


if __name__ == "__main__":
    main()
