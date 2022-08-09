import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

from util.h5_util import load_h5_file


def normalize(v, new_min = 0, new_max = 1):
    v_min, v_max = torch.min(v), torch.max(v)
    return (v - v_min)/(v_max - v_min) * (new_max - new_min) + new_min


def save_fig(fig_path: str, city: str, uq_method: str, filename: str):
    file_path = Path(os.path.join(fig_path, city))
    file_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(os.path.join(file_path, f"{uq_method}_{filename}.png"),
                dpi=300, format="png", bbox_inches='tight')
    print(f"Saved figure to {file_path}.")


def outlier_stats(out):
    samp, p_i, p_j, _ = tuple(out.shape)
    tot, pix_tot = samp * p_i * p_j, p_i * p_j

    print("### Outlier stats ###")

    ov, ov_pct = out[..., 0].sum(), out[..., 0].sum()/tot
    print(f"Total outliers by vol ch: {ov}/({samp}*{p_i}*{p_j}) or {(ov_pct*100):.2f}%.")

    os, os_pct = out[..., 1].sum(), out[..., 1].sum()/tot
    print(f"Total outliers by speed ch: {os}/({samp}*{p_i}*{p_j}) or {(os_pct*100):.2f}%.")

    op, op_pct = out[..., 2].sum(), out[..., 2].sum()/tot
    print(f"Total outliers by pixel: {op}/({samp}*{p_i}*{p_j}) or {(op_pct*100):.2f}%.")

    om = out[..., 2].sum(dim=0).max() # max outlier counts across sample dim for pixels
    omc = (out[..., 2].sum(dim=0) == om).sum()
    omc_pct = omc / tot
    print(f"""Pixels with max. outlier count by sample: {omc} pixels or
                 {(omc_pct*100):.2f}% with {om}/{samp} outliers.""")

    osamp = out[..., 2].sum(dim=(1,2)).to(torch.float32)
    osamp_m, osamp_std = int(osamp.mean().ceil()), int(osamp.std().ceil())
    osamp_pct_m, osamp_pct_std = osamp_m / pix_tot, osamp_std / pix_tot
    print(f"""Avg. pixel outlier count by sample: {osamp_m} +/- {osamp_std} of ({p_i}*{p_j})
                 or {(osamp_pct_m*100):.2f} +/- {(osamp_pct_std*100):.2f}%.""")

    osmax, osmin = osamp.argmax(), osamp.argmin()
    osma, osma_pct = int(osamp[osmax].item()), osamp[osmax] / pix_tot
    osmi, osmi_pct = int(osamp[osmin].item()), osamp[osmin] / pix_tot
    print(f"""Sample with most pixel outliers: test sample {osmax.item()}
                 with {osma}/({p_i}*{p_j}) outliers or {(osma_pct*100):.2f}%.""")
    print(f"""Sample with least pixel outliers: test sample {osmin.item()}
                 with {osmi}/({p_i}*{p_j}) outliers or {(osmi_pct*100):.2f}%.""")


# =============================================================================
# STATIC VALUES
# =============================================================================
map_path = "./data/raw"
fig_path = "./figures/out_detect"
base_path = "./results/out_detect"

city = "BARCELONA" # BANGKOK, BARCELONA
uq_method = "ensemble" # bnorm

out_b = {"ob01": 0.1, "ob025": 0.25, "ob05": 0.5, "ob075": 0.75, "ob1": 1, "ob2": 2}
ob = "ob1"


# =============================================================================
# Outliers whole city (vol, speed, pixel); mean over samp, fixed samp
# =============================================================================

out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
base_map = load_h5_file(os.path.join(map_path, city, f"{city}_static.h5"))[0]

# Mean outlier percentage per pixel across samp dim
cmap = plt.get_cmap("OrRd")
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = np.linspace(0, 1, cmap.N) # add transparency gradient
my_cmap = ListedColormap(my_cmap)

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
lab = ["vol", "speed", "pixel"]
for i, ax in enumerate(axes.flat):
    data = out[..., i].mean(dim=0)
    ax.imshow(base_map, cmap="gray_r", vmin=0, vmax=255)
    im = ax.imshow(data, cmap=my_cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"out {lab[i]}", fontsize="small")
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             "Mean outlier perc over test samp", fontsize="small")
fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=15, pad=0.03, shrink=0.75)

save_fig(fig_path, city, uq_method, f"mean_samp_{ob}")

# Outlier per pixel for fixed samp idx
samp = 50 # fixed
cmap = ListedColormap(["None", "red"])
# cmap = mpl.cm.get_cmap("OrRd", 2)
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
lab = ["vol", "speed", "pixel"]
for i, ax in enumerate(axes.flat):
    data = out[samp, :, :, i]
    ax.imshow(base_map, cmap="gray_r", vmin=0, vmax=255)
    ax.imshow(data, cmap=cmap, alpha=0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"out {lab[i]}", fontsize="small")
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Outliers for fixed test samp {samp}", fontsize="small")

save_fig(fig_path, city, uq_method, f"samp_{samp}_{ob}")


# =============================================================================
# Outliers crop city (vol, speed, pixel); fixed samp
# =============================================================================

out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))

# Outlier per pixel for fixed samp idx and fixed city crop
samp = 89
r = 390 # center pixel height (y-axis T to B)
c = 210 # center pixel width (x-axis L to R)
num = 30 # display: center +- num

cmap = ListedColormap(["None", "red"])
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
lab = ["vol", "speed", "pixel"]
for o, ax in enumerate(axes.flat):

    blup = torch.empty((2*10*num, 2*10*num))
    for i in range(-10*num, 10*num):
        for j in range(-10*num, 10*num):
            if r+i//10 >= 495 or c+j//10 >= 436:
                continue
            blup[i+10*num, j+10*num] = out[samp, r+i//10, c+j//10, o]

    im = ax.imshow(blup, cmap=cmap)
    ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.8, vmin=0, vmax=255)
    ax.set_title(f"out {lab[o]}", fontsize="small")
    # ax.set_xticklabels(np.arange(c-num, c+num+1, 2))
    # ax.set_xticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
    # ax.set_yticklabels(np.arange(r-num, r+num+1, 2))
    # ax.set_yticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Crop ({r-num}:{r+num}, {c-num}:{c+num}), " +
             f"Outliers for fixed test samp {samp}", fontsize="small")
# fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=15, pad=0.03, shrink=0.8)

save_fig(fig_path, city, uq_method, f"samp_{samp}_{ob}_crop_{r}_{c}")


# =============================================================================
# ANIMATION: Outliers whole city (vol, speed, pixel); over samp dim
# =============================================================================
# takes ~4m to run

out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
base_map = load_h5_file(os.path.join(map_path, city, f"{city}_static.h5"))[0]

cmap = ListedColormap(["None", "red"])
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
lab = ["vol", "speed", "pixel"]

def update(samp):
    for i, ax in enumerate(axes.flat):
        data = out[samp, :, :, i]
        ax.imshow(base_map, cmap="gray_r", vmin=0, vmax=255)
        ax.imshow(data, cmap=cmap, alpha=0.5)
        ax.set_title(f"out {lab[i]}", fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
                 f"Outliers for test samp {samp}", fontsize="small")

ani = FuncAnimation(fig, update, frames=out.shape[0], interval=200)
path = os.path.join(fig_path, city, f"{uq_method}_by_samp_{ob}.mp4")

# ani.save(path, writer=PillowWriter(fps=3)) # .gif
ani.save(path, fps=3) # .mp4


# =============================================================================
# ANIMATION: Outliers crop city (vol, speed, pixel); over samp dim
# =============================================================================
# takes ~15m to run

out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))

r = 390 # center pixel height (y-axis T to B)
c = 210 # center pixel width (x-axis L to R)
num = 30 # display: center +- num

cmap = ListedColormap(["None", "red"])
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
lab = ["vol", "speed", "pixel"]

def update(samp):
    for o, ax in enumerate(axes.flat):

        blup = torch.empty((2*10*num, 2*10*num))
        for i in range(-10*num, 10*num):
            for j in range(-10*num, 10*num):
                if r+i//10 >= 495 or c+j//10 >= 436:
                    continue
                blup[i+10*num, j+10*num] = out[samp, r+i//10, c+j//10, o]

        ax.imshow(blup, cmap=cmap)
        ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)],
                  cmap='gray_r', alpha=0.8, vmin=0, vmax=255)
        ax.set_title(f"out {lab[o]}", fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
                 f"Crop ({r-num}:{r+num}, {c-num}:{c+num}), " +
                 f"Outliers for test samp {samp}", fontsize="small")

ani = FuncAnimation(fig, update, frames=out.shape[0], interval=200)
path = os.path.join(fig_path, city, f"{uq_method}_by_samp_{ob}_crop_{r}_{c}.mp4")

# ani.save(path, writer=PillowWriter(fps=3)) # .gif
ani.save(path, fps=3) # .mp4


# =============================================================================
# ANIMATION: Outliers whole city per channel (8); over samp dim
# =============================================================================
# takes ~8m to run

from data.data_layout import channel_labels as ch_lab

pval = load_h5_file(os.path.join(base_path, city, f"pval_{uq_method}.h5"))
base_map = load_h5_file(os.path.join(map_path, city, f"{city}_static.h5"))[0]

out_bound = out_b[ob]/100
lab = ["volume_NW", "volume_NE", "speed_NW", "speed_NE",
       "volume_SW", "volume_SE", "speed_SW", "speed_SE"]
ch_id = [ch_lab.index(ch) for ch in lab] # change order of ch for better readability

cmap = ListedColormap(["None", "red"])
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

def update(samp):
    for i, ax in enumerate(axes.flat):
        data = pval[samp, :, :, ch_id[i]] <= out_bound # outliers per channel
        ax.imshow(base_map, cmap="gray_r", vmin=0, vmax=255)
        ax.imshow(data, cmap=cmap, alpha=0.5)
        ax.set_title(f"{lab[i]}", fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
                 f"Outliers for test samp {samp}", fontsize="small")

ani = FuncAnimation(fig, update, frames=pval.shape[0], interval=200)
path = os.path.join(fig_path, city, f"{uq_method}_by_samp_{ob}_per_ch.mp4")

# ani.save(path, writer=PillowWriter(fps=3)) # .gif
ani.save(path, fps=3) # .mp4


# =============================================================================
# Total pixel outlier counts whole city (vol, speed, pixel) vs. samp dim
# =============================================================================

out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
lab = ["vol", "speed", "pixel"]
for i, ax in enumerate(axes.flat):
    data = out[..., i].sum(dim=(1,2))
    ax.plot(data, ls="-", lw=0.7, marker=".", color="black", mec="black", mfc="red")
    ax.set_title(f"out {lab[i]}", fontsize="small")
    # ax.set_ylabel("Outlier count (sum over pixels)")
    # ax.set_xlabel("Test sample")
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             "Total outlier count (sum over pixels) vs. test samples", fontsize="small")

save_fig(fig_path, city, uq_method, f"out_counts_{ob}")


# =============================================================================
# P-values & outlier bounds vs. samp dim; fixed pixel
# =============================================================================

from scipy.stats import combine_pvalues

# pred_tr = load_h5_file(os.path.join(base_path, city, f"pred_tr_{uq_method}.h5"))
pval = load_h5_file(os.path.join(base_path, city, f"pval_{uq_method}.h5"))

# find pixel: (out[..., 2].sum(dim=0) == 90).nonzero(as_tuple=False)
pix_h, pix_w = 386, 191 # fixed pixel
samp_c = 90 # sample count, not fixed sample!
pval_pix = pval[:, pix_h, pix_w, :]

agg_pval = torch.empty(size=(samp_c, 2), dtype=torch.float32)
for samp in range(samp_c):
    agg_pval[samp, 0] = 1 - combine_pvalues(pval_pix[samp, [0, 2, 4, 6]].clamp(min=1e-10), method="fisher")[1]
    agg_pval[samp, 1] = 1 - combine_pvalues(pval_pix[samp, [1, 3, 5, 7]].clamp(min=1e-10), method="fisher")[1]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
lab = ["vol", "speed"]
for i, ax in enumerate(axes.flat):
    data = agg_pval[:, i]
    ax.scatter(range(len(data)), data, label="1 - p-value (Fisher)", marker=".", color="black")
    # ax.plot(data, label="1 - p-value (Fisher)", color="black")
    ax.set_title(f"pval {lab[i]}", fontsize="small")

    out_bound = 0.01
    ax.axhline(1-out_bound, color="sienna", linestyle=":", label="outlier bound 1%")
    ax.text(x=0, y=1-out_bound-5e-4, s=rf"{(agg_pval[:, i] >= 1-out_bound).sum().item()}$\uparrow$", color="sienna", fontweight="bold")

    out_bound = 0.005
    ax.axhline(1-out_bound, color="orange", linestyle=":", label="outlier bound 0.5%")
    ax.text(x=0, y=1-out_bound-5e-4, s=rf"{(agg_pval[:, i] >= 1-out_bound).sum().item()}$\uparrow$", color="orange", fontweight="bold")

    out_bound = 0.001
    ax.axhline(1-out_bound, color="red", linestyle=":", label="outlier bound 0.1%")
    ax.text(x=0, y=1-out_bound-5e-4, s=rf"{(agg_pval[:, i] >= 1-out_bound).sum().item()}$\uparrow$", color="red", fontweight="bold")

    ax.set_ylabel("1 - p-value")
    ax.set_ylim(0.98, 1.0)
    ax.legend()
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, Pixel ({pix_h}, {pix_w}), " +
             "P-values (Fisher) & outlier bounds vs. test samples", fontsize="small")

save_fig(fig_path, city, uq_method, f"pval_ob_pix_{pix_h}_{pix_w}")


# =============================================================================
# Total (pixel+samp) outlier counts whole city (vol, speed, pixel) vs. out bounds
# =============================================================================

ob_str = list(out_b.keys())
out_ob = torch.empty(size=(len(ob_str), 3), dtype=torch.float32)

for i, ob in enumerate(ob_str):
    out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
    out_ob[i, 0] = (out[..., 0].sum(dim=(0,1,2))/(90*495*436))*100
    out_ob[i, 1] = (out[..., 1].sum(dim=(0,1,2))/(90*495*436))*100
    out_ob[i, 2] = (out[..., 2].sum(dim=(0,1,2))/(90*495*436))*100
    del out

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
lab = ["vol", "speed", "pixel"]
for i, ax in enumerate(axes.flat):
    data = out_ob[:, i]
    ax.plot(data, ls="-", lw=1, marker="o", color="black", mec="black", mfc="red")
    ax.set_title(f"out {lab[i]}", fontsize="small")
    ax.set_ylabel("Outlier portion (in %)")
    ax.set_xlabel("Outlier bound (in %)")
    ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['0.1', '0.25', '0.5', '0.75', '1', '2'])
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, " +
             "Perc of total outlier count (sum over pixels + samples) vs. outlier bounds", fontsize="small")

save_fig(fig_path, city, uq_method, "out_perc_vs_ob")
