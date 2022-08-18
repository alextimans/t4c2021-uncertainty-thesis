import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

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


# =============================================================================
# STATIC VALUES
# =============================================================================
map_path = "./data/raw"
fig_path = "./figures/test_100/test_100_1"
base_path = "./results/test_100/test_100_1/h5_files"

city = "MOSCOW" # BANGKOK, BARCELONA, MOSCOW
uq_method = "point" # bnorm, combo, point, ensemble, patches, tta
m = "False" # Masked values: True, False

sc_idx = {"mean_gt": 0, "mean_pred": 1, "mean_unc": 2, "mean_mse": 3,
             "std_gt": 4, "std_pred": 5, "std_unc": 6, "std_mse": 7,
             "pi_width": 8, "cover": 9, "ence": 10, "cov": 11,
             "corr": 12, "sp_corr": 13}
m_idx = {"True": "_mask", "False": ""}

# DEFINE DICT TO FILTER FOR SPEED OR VOL CHANNELS (SEPARATE PLOTS)

# =============================================================================
# Mean uncertainty vs. mean MSE
# =============================================================================
scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))
unc = torch.sum(scores[sc_idx["mean_unc"], ...], dim=-1)#[150:300, 150:300] #[150:300, 150:300], [220:250, 230:260]
mse = torch.sum(scores[sc_idx["mean_mse"], ...], dim=-1)#[150:300, 150:300]

cmap = plt.get_cmap("OrRd")
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = np.linspace(0, 1, cmap.N) # add transparency gradient
my_cmap = ListedColormap(my_cmap)

fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(4, 2.8),
                                    gridspec_kw={"width_ratios":[1, 1, 0.05]})
fig.subplots_adjust(wspace=0.3)
# norm = mpl.colors.Normalize()
ax1.set_title("MSE", fontsize="small")
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax2.set_title("Uncertainty", fontsize="small")
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

im1 = ax1.imshow(normalize(torch.log(mse)), cmap=my_cmap)
im2 = ax2.imshow(normalize(torch.log(unc)), cmap=my_cmap)
cax.set_axes_locator(InsetPosition(ax2, [1.05, 0, 0.05, 1])) # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
fig.colorbar(im1, cax=cax, ax=[ax1, ax2])
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, Mask: {m},\n " +
             "Mean (all ch) log-norm MSE vs. unc", fontsize="small")

save_fig(fig_path, city, uq_method, "mse_unc")


# TODO FROM HERE ON

### Mean uncertainty - RMSE (1 plot)

diff = torch.abs(torch.sqrt(mse) - unc) #[150:300, 150:300], [220:250, 230:260]

fig, ax = plt.subplots()
norm = mpl.colors.Normalize()
im = ax.imshow(norm(diff), cmap=cmap)
plt.colorbar(im, location="right", aspect=40, pad=0.02)
ax.set_title("Difference uncertainty to RMSE (normalized)")


# =============================================================================
# Pearson corr. vs. ENCE
# =============================================================================
corr = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "calibration", stride), dim=-1)
ence = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "ence_scores", stride), dim=-1)

fig, ax = plt.subplots()
im = ax.imshow(corr, cmap=cmap)
plt.colorbar(im, location="right", aspect=40, pad=0.02)
ax.set_title("Pearson correlation")

fig, ax = plt.subplots()
norm = mpl.colors.Normalize()
im = ax.imshow(torch.log(ence), cmap=cmap)
plt.colorbar(im, location="right", aspect=40, pad=0.02)
ax.set_title("ENCE")

fig, ax = plt.subplots()
ax.hist(corr)

fig, ax = plt.subplots()
ax.hist(ence)


# =============================================================================
# High res map overlay
# =============================================================================
static = load_h5_file(f"{base_path}/{city}_map_high_res.h5")
data = get_file_from_path(base_path, uq_method, city, channels, "mean_gt_pred_unc_err", stride)
#[150:300, 150:300], [220:250, 230:260]
dat = torch.log(torch.sum(data[3, ...], dim=-1)) # mse log of sum of speed channels
dat = torch.mean(data[2, ...], dim=-1) # mean uncertainty

r = 230 # center pixel height
c = 240 # center pixel width
num = 10 # display: center +- num

fig, ax = plt.subplots()
ax.set_title(f"Uncertainty in ({r-num}:{r+num}, {c-num}:{c+num})")

# blow-up 10m -> 100m
blup = torch.empty((2*10*num, 2*10*num))
for i in range(-10*num, 10*num):
    for j in range(-10*num, 10*num):
        if r+i//10 >= 495 or c+j//10 >= 436:
            continue
        blup[i+10*num, j+10*num] = dat[r+i//10, c+j//10]

im = ax.imshow(blup, cmap=cmap)
fig.colorbar(im, ax=ax)
ax.imshow(static[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.5, vmin=0, vmax=255)
ax.set_xticklabels(np.arange(c-num, c+num+1, 2))
ax.set_xticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
ax.set_yticklabels(np.arange(r-num, r+num+1, 2))
ax.set_yticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))


# =============================================================================
# Mean PI width vs. std of ground truths
# =============================================================================
# Kind of the same statement as mean uncertainty vs. MSE but with different proxies ?

data = get_file_from_path(base_path, uq_method, city, channels, "std_gt_pred_unc_err", stride)
std_gt = torch.mean(data[0, ...], dim=-1) #[150:300, 150:300], [220:250, 230:260]
pi_width = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "pi_width", stride), dim=-1)

fig, (ax1, ax2, cax) = plt.subplots(1, 3, gridspec_kw={"width_ratios":[1, 1, 0.05]})
fig.subplots_adjust(wspace=0.05)
norm = mpl.colors.Normalize()
im1 = ax1.imshow(norm(std_gt), cmap=cmap)
im2 = ax2.imshow(norm(pi_width), cmap=cmap)

cax.set_axes_locator(InsetPosition(ax2, [1.05, 0, 0.05, 1])) # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
fig.colorbar(im1, cax=cax, ax=[ax1, ax2])

ax1.set_title("Ground truth std (normalized)", fontsize="medium")
ax2.set_title("Mean PI width (normalized)", fontsize="medium")
ax2.get_yaxis().set_visible(False)


# =============================================================================
# Data sparseness
# =============================================================================
data = get_file_from_path(base_path, uq_method, city, "vol", "mean_gt_pred_unc_err", stride)

torch.mean(data[2, data[0, ...]==0]) # mean uncertainty for zero gt
torch.mean(data[2, data[0, ...]!=0]) # mean uncertainty for non zero gt
torch.mean(data[3, data[0, ...]==0]) # mean mse for zero gt
torch.mean(data[3, data[0, ...]!=0]) # mean mse for non zero gt

mean_gt = torch.mean(data[0, ...], dim=-1)
fig, ax = plt.subplots()
im = ax.spy(mean_gt, origin="lower")
plt.colorbar(im, location="right", aspect=40, pad=0.02)
ax.set_title("Nonzero ground truths")
ax.xaxis.tick_bottom()


# =============================================================================
# Diff uncertainty - RMSE per channel + pearson corr
# =============================================================================
data = get_file_from_path(base_path, uq_method, city, channels, "mean_gt_pred_unc_err", stride)
#[150:300, 150:300], [220:250, 230:260]
diff = torch.abs(torch.sqrt(data[3, ...]) -  data[2, ...])
corr = get_file_from_path(base_path, uq_method, city, channels, "calibration", stride)

fig, axes = plt.subplots(2, 2, figsize=(7,7))
fig.subplots_adjust(wspace=0.05, hspace=0.1)
im = plt.cm.ScalarMappable(cmap=cmap)
for i, ax in enumerate(axes.flat):
    if i == 0:
        ax.get_xaxis().set_visible(False)
    elif i == 1:
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
    elif i == 3:
        ax.get_yaxis().set_visible(False)
    ax.imshow(diff[:, :, i], cmap=cmap)
    ax.set_title(f"Speed channel {i}, " + r"$\rho$=" + "{:.2f}".format(torch.mean(corr[:, :, i]).item()))
fig.suptitle("Difference uncertainty - RMSE", y=0.94, fontweight="semibold")
fig.colorbar(im, ax=axes.ravel().tolist(), location="right", aspect=40, pad=0.02, shrink=0.98)


# =============================================================================
# Mean PI widths vs. coverage
# =============================================================================
cov = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "coverage", stride), dim=-1).flatten()
pi = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "pi_width", stride), dim=-1).flatten()

bins = 100
bin_val = torch.empty(size=(bins, 2), dtype=torch.float32)
samp_per_bin = int(pi.shape[0] / bins)
sort_idx = torch.sort(pi, descending=False)[1]

for cbin in range(bins):
    idx = sort_idx[(cbin * samp_per_bin):(cbin * samp_per_bin + samp_per_bin)]
    bin_val[cbin, 0] = torch.mean(pi[idx])
    bin_val[cbin, 1] = torch.mean(cov[idx])

fig, ax = plt.subplots()
ax.plot(bin_val[:, 0], bin_val[:, 1], color="darkorange")
ax.set_xscale("log")
ax.set_ylim(0.87, 0.93)
ax.set_xlabel("Binned mean PI widths (log scale)")
ax.set_ylabel("Coverage")
ax.set_title("Empirical and nominal coverage for mean PI widths")
ax.axhline(y=0.9, color="grey", ls=":")


# =============================================================================
# FOR t4c PAPER
# =============================================================================

# =============================================================================
# All metrics in a row (unmasked)
# =============================================================================
means = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "mean_gt_pred_unc_err", stride), dim=-1)
pi = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "pi_width", stride), dim=-1)
corr = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "calibration", stride), dim=-1)
ence = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "ence_scores", stride), dim=-1)
data = torch.stack((normalize(torch.log(means[0]+1e-5)),
                    normalize(torch.log(means[2])),
                    normalize(torch.log(means[3])),
                    normalize(torch.log(pi)),
                    corr,
                    normalize(torch.log(ence))), dim=0)
titles = ["Ground truth", "MSE", "Uncertainty", "Mean PI width", r"Correlation $\rho$", "ENCE"]

cmap="coolwarm" # coolwarm
fig, axes = plt.subplots(1, 6, figsize=(10, 2.1))
fig.subplots_adjust(wspace=0.05)
for i, ax in enumerate(axes.flat):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(data[i, ...], cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(titles[i], fontsize="small")
fig.suptitle(f"{city.capitalize()}", y=0.95, x=0.45, fontweight="semibold")
fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=15, pad=0.01, shrink=0.7)

save_fig(f"all_metrics_unmasked_{city}", base_path, uq_method, city, stride)

# =============================================================================
# Ground truth masked correlations
# =============================================================================
means = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "mean_gt_pred_unc_err", stride), dim=-1)
corr = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "calibration", stride), dim=-1)
gt_zero = means[0, ...] == 0
gt_nonzero = means[0, ...] != 0

# masked for nonzero gt
cbar_kws={"location": "right", "aspect": 40, "pad": 0.02}
ax = sns.heatmap(corr.numpy(), mask=gt_zero.numpy(), cmap="coolwarm",
                 vmin=-1, vmax=1, cbar_kws=cbar_kws, edgecolors="black")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title(r"Correlation $\rho$" + f" for non-zero ground truths, {city.capitalize()}")
for _, spine in ax.spines.items():
    spine.set_visible(True)

save_fig(f"corr_nonzero_{city}", base_path, uq_method, city, stride)

# masked for zero gt
cbar_kws={"location": "right", "aspect": 40, "pad": 0.02}
ax = sns.heatmap(corr.numpy(), mask=gt_nonzero.numpy(), cmap="coolwarm",
                 vmin=-1, vmax=1, cbar_kws=cbar_kws, edgecolors="black")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title(r"Correlation $\rho$" + f" for zero ground truths, {city.capitalize()}")
for _, spine in ax.spines.items():
    spine.set_visible(True)

save_fig(f"corr_zero_{city}", base_path, uq_method, city, stride)

# no masking
cbar_kws={"location": "right", "aspect": 40, "pad": 0.02}
ax = sns.heatmap(corr.numpy(), cmap="coolwarm",
                 vmin=-1, vmax=1, cbar_kws=cbar_kws, edgecolors="black")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title(r"Correlation $\rho$" + f" for all ground truths, {city.capitalize()}")
for _, spine in ax.spines.items():
    spine.set_visible(True)

save_fig(f"corr_all_{city}", base_path, uq_method, city, stride)

# =============================================================================
# Pearson correlation histograms zero vs. non-zero ground truths
# =============================================================================
means = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "mean_gt_pred_unc_err", stride), dim=-1)
corr = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "calibration", stride), dim=-1)
gt_zero = means[0, ...] == 0
gt_nonzero = means[0, ...] != 0

fig, ax = plt.subplots(figsize=(4, 2.8))
ax.hist(corr[gt_zero].numpy(), bins=40, alpha=0.6, range=(-1,1),
        color="red", label="Zero ground truth", density=True)
ax.hist(corr[gt_nonzero].numpy(), bins=40, alpha=0.6, range=(-1,1),
        color="blue", label="Non-zero ground truth", density=True)
ax.legend(loc="upper left", fontsize="medium") # large
ax.set_ylabel("Density")
ax.set_xlabel(r"Correlation $\rho$")
ax.set_title(f"Correlation histograms for {city.capitalize()}", fontsize="medium")
ax.grid()

save_fig(f"corr_hist2_{city}", base_path, uq_method, city, stride)

# =============================================================================
# Anecdotal outcrop metric with high res map overlay
# =============================================================================
static = load_h5_file(f"{base_path}/{city}_map_high_res.h5")
data = get_file_from_path(base_path, uq_method, city, channels, "mean_gt_pred_unc_err", stride)
#[150:300, 150:300], [220:250, 230:260]
dat = normalize(torch.log(torch.mean(data[2, ...], dim=-1)))
metric, ID = "unc", 2
r = 230 # center pixel height
c = 240 # center pixel width
num = 10 # display: center +- num

# ANTWERP interesting points
# r=235; c=242; num=10
# r=212; c=178; num=10
# r=197; c=220; num=10

# BANGKOK interesting points
# r=285; c=240; num=15
# r=265; c=100; num=15
# r=300; c=293; num=15

blup = torch.empty((2*10*num, 2*10*num))
for i in range(-10*num, 10*num):
    for j in range(-10*num, 10*num):
        if r+i//10 >= 495 or c+j//10 >= 436:
            continue
        blup[i+10*num, j+10*num] = dat[r+i//10, c+j//10]

fig, ax = plt.subplots()
im = ax.imshow(blup, cmap="OrRd")
ax.imshow(static[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.5, vmin=0, vmax=255)
fig.colorbar(im, ax=ax, aspect=40, pad=0.02)

ax.set_title(f"Uncertainty in ({r-num}:{r+num}, {c-num}:{c+num})")
ax.set_xticklabels(np.arange(c-num, c+num+1, 2))
ax.set_xticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
ax.set_yticklabels(np.arange(r-num, r+num+1, 2))
ax.set_yticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

save_fig(f"zoom_{metric}_{ID}_{city}", base_path, uq_method, city, stride)

# to search and find interesting points
# d = dat[285:325, 280:320]
d = dat[:, :]
fig, ax = plt.subplots()
im = ax.imshow(d, cmap="OrRd")
save_fig(f"gt_{city}_map", base_path, uq_method, city, stride)

# =============================================================================
# Anecdotal outcrop metric multiple at once
# =============================================================================
static = load_h5_file(f"{base_path}/{city}_map_high_res.h5")
data = get_file_from_path(base_path, uq_method, city, channels, "mean_gt_pred_unc_err", stride)
dat = normalize(torch.log(torch.mean(data[2, ...], dim=-1)))

# r_list, c_list, num = [235, 212, 197], [242, 178, 220], 10 # Antwerp
r_list, c_list, num = [285, 265, 300], [240, 100, 293], 15 # Bangkok
metric = "unc"

fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
fig.subplots_adjust(wspace=0.05)
for i, ax in enumerate(axes.flat):
    r, c = r_list[i], c_list[i]

    blup = torch.empty((2*10*num, 2*10*num))
    for i in range(-10*num, 10*num):
        for j in range(-10*num, 10*num):
            if r+i//10 >= 495 or c+j//10 >= 436:
                continue
            blup[i+10*num, j+10*num] = dat[r+i//10, c+j//10]

    im = ax.imshow(blup, cmap="OrRd", vmin=0.3, vmax=1)
    ax.imshow(static[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.5, vmin=0, vmax=255)
    ax.set_title(f"Crop ({r-num}:{r+num}, {c-num}:{c+num})", fontsize="small")
    ax.set_xticklabels(np.arange(c-num, c+num+1, 2))
    ax.set_xticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
    ax.set_yticklabels(np.arange(r-num, r+num+1, 2))
    ax.set_yticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig.suptitle(f"{city.capitalize()}", y=0.95, x=0.45, fontweight="semibold")
fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=30, pad=0.01, shrink=0.8)

save_fig(f"zoom_all_{metric}_{city}", base_path, uq_method, city, stride)


# =============================================================================
# full city plot
# =============================================================================
data = get_file_from_path(base_path, uq_method, city, channels, "mean_gt_pred_unc_err", stride)
#data = load_h5_file("./data/test_tta_unet2/scores_tta.h5")
#dat = normalize(torch.log(torch.mean(data[8, :, :, [0,2,4,6]], dim=-1)+1e-5))
dat = normalize(torch.log(torch.mean(data[0, ...], dim=-1)+1e-5))

fig, ax = plt.subplots()
im = ax.imshow(dat, cmap="OrRd")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

save_fig(f"gt_{city}_map", base_path, uq_method, city, stride)

# =============================================================================
# full city plot my data
# =============================================================================
data = load_h5_file("./data/test_tta_unet2/scores_tta.h5")
dat = normalize(torch.log(torch.mean(data[0, :, :, [1,3,5,7]], dim=-1)+1e-5))
# dat = normalize(torch.log(torch.mean(data[2, :, :, [1,3,5,7]], dim=-1)))
# dat=torch.mean(data[8, :, :, [1,3,5,7]], dim=-1)

fig, ax = plt.subplots()
im = ax.imshow(dat, cmap="OrRd")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

meta = load_h5_file("./data/raw/BERLIN/BERLIN_test_additional_temporal.h5")
quant = load_h5_file("./data/test_tta_unet2/calib_quant_90_tta_unet2.h5")


# =============================================================================
# histogram empirical coverage vs. beta distr.
# =============================================================================
from scipy.stats import beta
n = 100; alpha=0.1 # calibration samples, coverage
a, b = n + 1 - np.floor((n+1)*alpha), np.floor((n+1)*alpha) # beta shape params
x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 500)

cov = torch.mean(get_file_from_path(base_path, uq_method, city, channels, "coverage", stride), dim=-1).flatten()

fig, ax = plt.subplots(figsize=(4, 2.8))
ax.hist(cov.numpy(), bins=30, alpha=0.6, range=(0.7,1),
        color="blue", density=True)
ax.plot(x, beta.pdf(x, a, b), color="red", label=f"Beta({int(a)},{int(b)})")
ax.set_xlim(0.7, 1)
ax.legend(loc="upper left", fontsize="medium") # large
ax.set_ylabel("Density")
ax.set_xlabel(r"Coverage")
ax.set_title(f"Coverage histogram for {city.capitalize()}", fontsize="medium")
ax.grid()

save_fig(f"cov_hist_{city}", base_path, uq_method, city, stride)
