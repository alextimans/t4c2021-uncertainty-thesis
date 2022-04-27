# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import itertools

offset_map = {"N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1), "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)}
layer_indices_from_offset = {v: i + 1 for i, v in enumerate(offset_map.values())}  # noqa

heading_list = ["NE", "SE", "SW", "NW"]
channel_labels = list(itertools.chain.from_iterable([[f"volume_{h}", f"speed_{h}"] for h in heading_list]))
static_channel_labels = ["base_map"] + [f"connectivity_{d}" for d in offset_map.keys()]

volume_channel_indices = [ch for ch, l in enumerate(channel_labels) if "volume" in l]
speed_channel_indices = [ch for ch, l in enumerate(channel_labels) if "speed" in l]

# Constants

# Used in eval.py
CITY_NAMES = ["ANTWERP", "BANGKOK", "BARCELONA", "BERLIN",
              "CHICAGO", "ISTANBUL", "MELBOURNE", "MOSCOW"]

# Used in dataset.py, eval.py
MAX_ONE_DAY_SMP_IDX = 264 # Last sample 22-24h (5m slot indices 264-288)
MAX_FILE_DAY_IDX = 288 # 288 x 5m slots per h5 file
TWO_HOURS = 24 # 12 * 2 x 5m slots

# Used in train_val_split.py
TRAIN_FILES = 126 # 0.7 * 180 h5 files per city and year
VAL_FILES = 27 # 0.15 * 180 h5 files per city and year
TEST_FILES = 27 # 0.15 * 180 h5 files per city and year
