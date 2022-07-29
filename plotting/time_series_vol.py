import glob
import matplotlib.pyplot as plt
import torch
import os
import datetime
import random

from util.h5_util import load_h5_file

city = "BANGKOK" # BANGKOK, BARCELONA, MOSCOW
date = "2019-03-25" # 2019-01-02 ... 2019-06-30
file = os.path.join("data", "raw", city, "train", f"{date}_{city}_8ch.h5")
data = load_h5_file(file)
print(f'Data: shape {data.shape} and type {data.dtype}')

# sum over vol channels and pixels
ts = torch.sum(data[..., [0, 2, 4, 6]], dim=(1,2,3))
#x_lab = [datetime.time(hour=t // 12, minute=t % 12 * 5).isoformat(timespec="minutes") for t in range(288, step=10)]

### Plot

plt.title(f"{city} {date} sum vol channels; max at {ts.argmax()}")
plt.plot(ts)

t = 100
datetime.time(hour=t // 12, minute=t % 12 * 5).isoformat(timespec="minutes")

### Multiple plot

s = 10
city = "MOSCOW" # BANGKOK, BARCELONA, MOSCOW
path = os.path.join("data", "raw", city, "train", f"*_{city}_8ch.h5")
files = sorted(glob.glob(path, recursive=True))
rand_idx = random.sample(range(len(files)), s)
file = [files[i] for i in rand_idx]

ts = torch.empty((288, s))
for i, f in enumerate(file):
    data = load_h5_file(f)
    ts[:, i] = torch.sum(data[..., [0, 2, 4, 6]], dim=(1,2,3))

for i in range(s):
    plt.plot(ts[:, i], label=file[i].split("/")[-1].split("_")[0])
plt.title(city)
plt.legend()
plt.show()
