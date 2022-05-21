import os
import numpy as np
import matplotlib.pyplot as plt


def convolve(loss: np.ndarray, k_size: int) -> np.ndarray:
    return np.convolve(loss, np.ones(k_size)/k_size, mode="valid")


def get_ma_loss(loss_path: str, epoch_count: int,
                ma_k_size: int, dataset: str = "val") -> np.ndarray:

    assert dataset in ["train", "val"]
    ds = "t" if dataset == "train" else "v"

    losses = []
    for epoch in range(epoch_count):
        path = os.path.join(loss_path, f"loss_{ds}_bybatch_{epoch}.txt")
        loss = np.loadtxt(path)
        losses.append(loss)

    loss = convolve(np.concatenate(losses), ma_k_size)

    return loss


def plot_loss_by_batch(loss_path: str, epoch_count: int,
                       ma_k_size: int, fig_save: bool = True,
                       fig_path: str = "", fig_name: str = "Figure"):

    dataset1, dataset2 = "train", "val"
    loss1 = get_ma_loss(loss_path, epoch_count, ma_k_size, dataset1)
    loss2 = get_ma_loss(loss_path, epoch_count, ma_k_size, dataset2)

    ep1 = len(loss1) // epoch_count
    ep2 = len(loss2) // epoch_count

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(loss1, color="red")
    ax1.axvline(x=ep1, color="black", ls="--")
    x_text, y_text = ep1 + ep1*0.05, ax1.get_ylim()[1]*0.95
    ax1.text(x_text, y_text, "Epoch")
    for ep in range(2, epoch_count):
        ax1.axvline(x=ep1*ep, color="black", ls="--")
    ax1.xaxis.set_major_formatter(lambda x, pos: str(int(x*1e-3)))
    ax1.set_xlabel("Batches (per 1000)")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title(f"{dataset1} loss")
    ax1.grid()

    ax2.plot(loss2, color="red")
    ax2.axvline(x=ep2, color="black", ls="--")
    x_text, y_text = ep2 + ep2*0.05, ax2.get_ylim()[1]*0.95
    ax2.text(x_text, y_text, "Epoch")
    for ep in range(2, epoch_count):
        ax2.axvline(x=ep2*ep, color="black", ls="--")
    ax2.xaxis.set_major_formatter(lambda x, pos: str(int(x*1e-3)))
    ax2.set_xlabel("Batches (per 1000)")
#    ax2.set_ylabel("MSE Loss")
    ax2.set_title(f"{dataset2} loss")
    ax2.grid()

    if fig_save:
        fig_path = os.path.join(fig_path, fig_name)
        plt.savefig(fig_path, dpi=300, format="png", bbox_inches='tight')
    

def plot_loss_by_epoch(loss_path: list, fig_save: bool = True,
                       fig_path: str = "", fig_name: str = "Figure",
                       sharey: bool = False):

    loss_t = np.loadtxt(os.path.join(loss_path, "loss_train.txt"))
    loss_v = np.loadtxt(os.path.join(loss_path, "loss_val.txt"))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=sharey)

    ax1.plot(loss_t, color="red", ls="-", marker="o", label="full data")
    ax1.set_ylabel("MSE Loss")
    ax1.set_xticks([0,1,2,3])
    ax1.set_xlabel("Epoch")
    ax1.set_title("Train loss (mean per epoch)")
    ax1.grid()
#    ax1.legend()

    ax2.plot(loss_v, color="red", ls="-", marker="o", label="full data")
#    ax2.set_ylabel("MSE Loss")
    ax2.set_xticks([0,1,2,3])
    ax2.set_xlabel("Epoch")
    ax2.set_title("Val loss (mean per epoch)")
    ax2.grid()
#    ax2.legend()

    if fig_save:
        fig_path = os.path.join(fig_path, fig_name)
        plt.savefig(fig_path, dpi=300, format="png", bbox_inches='tight')


plot_loss_by_batch(loss_path="./checkpoints/unet_1",
                   epoch_count=4,
                   ma_k_size=300,
                   fig_save=True,
                   fig_path="./figures",
                   fig_name="unet_1 train+val loss by batch")

plot_loss_by_epoch(loss_path="./checkpoints/unet_1",
                   fig_save=True,
                   fig_path="./figures",
                   fig_name="unet_1 train+val loss by epoch",
                   sharey=True)


"""
path_ep0_v_1 = "./checkpoints/unet_1/loss_v_bybatch_0.txt"
path_ep1_v_1 = "./checkpoints/unet_1/loss_v_bybatch_1.txt"

loss_ep0_v_1 = np.loadtxt(path_ep0_v_1)
loss_ep1_v_1 = np.loadtxt(path_ep1_v_1)

loss_val_1 = np.concatenate([loss_ep0_v_1, loss_ep1_v_1])
loss_val_1_ma = convolve(loss_val_1, 5000)

path_ep0_v_2 = "./checkpoints/unet_2/loss_v_bybatch_0.txt"
path_ep1_v_2 = "./checkpoints/unet_2/loss_v_bybatch_1.txt"

loss_ep0_v_2 = np.loadtxt(path_ep0_v_2)
loss_ep1_v_2 = np.loadtxt(path_ep1_v_2)

loss_val_2 = np.concatenate([loss_ep0_v_2, loss_ep1_v_2])
loss_val_2_ma = convolve(loss_val_2, 5000)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.plot(loss_val_1_ma, color="red")
ax1.axvline(x=23328, color="black", ls="--")
ax1.text(23328+1000, 108, "Epoch")
ax1.set_xlabel("Batches")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Val loss on full dataset")
ax1.grid()

ax2.plot(loss_val_2_ma, color="red")
ax2.axvline(x=15552, color="black", ls="--")
ax2.text(15552+1000, 108, "Epoch")
ax2.set_xlabel("Batches")
ax2.set_title("Val loss on partial dataset")
ax2.grid()
"""


loss_1 = np.loadtxt("./checkpoints/unet_1_val_by_city/loss_v_bybatch_ANTWERP.txt")
loss_2 = np.loadtxt("./checkpoints/unet_1_val_by_city/loss_v_bybatch_BANGKOK.txt")
loss_3 = np.loadtxt("./checkpoints/unet_1_val_by_city/loss_v_bybatch_BARCELONA.txt")
loss_4 = np.loadtxt("./checkpoints/unet_1_val_by_city/loss_v_bybatch_MOSCOW.txt")

loss_1 = convolve(loss_1, 50)
loss_2 = convolve(loss_2, 50)
loss_3 = convolve(loss_3, 50)
loss_4 = convolve(loss_4, 50)

#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig, ax1 = plt.subplots()

ax1.plot(loss_1, color="blue", label="ANTWERP")
ax1.plot(loss_2, color="green", label="BANGKOK")
ax1.plot(loss_3, color="red", label="BARCELONA")
ax1.plot(loss_4, color="purple", label="MOSCOW")

#ax1.xaxis.set_major_formatter(lambda x, pos: str(int(x*1e-3)))
ax1.set_xlabel("Batches")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Val loss for each city (moving avg. over 50 batches)")
ax1.grid()
ax1.legend()

fig_path = os.path.join("./figures", "val loss each city")
plt.savefig(fig_path, dpi=300, format="png", bbox_inches='tight')
