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


def plot_loss_by_batch(loss_path: list, epoch_counts: list,
                       ma_k_size: int, dataset: str = "val",
                       fig_save: bool = True, fig_path: str = "",
                       fig_name: str = "Figure"):

    loss1 = get_ma_loss(loss_path[0], epoch_counts[0], ma_k_size, dataset)
    loss2 = get_ma_loss(loss_path[1], epoch_counts[1], ma_k_size, dataset)

    ep1 = len(loss1) // epoch_counts[0]
    ep2 = len(loss2) // epoch_counts[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(loss1, color="red")
    ax1.axvline(x=ep1, color="black", ls="--")
    x_text, y_text = ep1 + ep1*0.05, ax1.get_ylim()[1]*0.99
    ax1.text(x_text, y_text, "Epoch")
    ax1.xaxis.set_major_formatter(lambda x, pos: str(int(x*1e-3)))
    ax1.set_xlabel("Batches (per 1000)")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title(f"{dataset} loss on full dataset")
    ax1.grid()

    ax2.plot(loss2, color="red")
    ax2.axvline(x=ep2, color="black", ls="--")
    x_text, y_text = ep2 + ep2*0.05, ax2.get_ylim()[1]*0.99
    ax2.text(x_text, y_text, "Epoch")
    ax2.xaxis.set_major_formatter(lambda x, pos: str(int(x*1e-3)))
    ax2.set_xlabel("Batches (per 1000)")
    ax2.set_title(f"{dataset} loss on partial dataset")
    ax2.grid()

    if fig_save:
        fig_path = os.path.join(fig_path, fig_name)
        plt.savefig(fig_path, dpi=300, format="png", bbox_inches='tight')
    

def plot_loss_by_epoch(loss_path: list, fig_save: bool = True,
                       fig_path: str = "", fig_name: str = "Figure"):

    loss_v_1 = np.loadtxt(os.path.join(loss_path[0], "loss_val.txt"))
    loss_v_2 = np.loadtxt(os.path.join(loss_path[1], "loss_val.txt"))
    loss_t_1 = np.loadtxt(os.path.join(loss_path[0], "loss_train.txt"))
    loss_t_2 = np.loadtxt(os.path.join(loss_path[1], "loss_train.txt"))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(loss_v_1, color="red", ls="-", marker="o", label="full data")
    ax1.plot(loss_v_2, color="green", ls="--", marker="o", label="partial data")
    ax1.set_ylabel("MSE Loss")
    ax1.set_xticks([0,1])
    ax1.set_xlabel("Epochs")
    ax1.set_title("Val loss")
    ax1.grid()
#    ax1.legend()

    ax2.plot(loss_t_1, color="red", ls="-", marker="o", label="full data")
    ax2.plot(loss_t_2, color="green", ls="--", marker="o", label="partial data")
#    ax2.set_ylabel("MSE Loss")
    ax2.set_xticks([0,1])
    ax2.set_xlabel("Epochs")
    ax2.set_title("Train loss")
    ax2.grid()
    ax2.legend()

    if fig_save:
        fig_path = os.path.join(fig_path, fig_name)
        plt.savefig(fig_path, dpi=300, format="png", bbox_inches='tight')


plot_loss_by_batch(loss_path=["./checkpoints/unet_1_range01",
                              "./checkpoints/unet_1_range01"],
                   epoch_counts=[2, 2],
                   ma_k_size=5000,
                   dataset="val",
                   fig_save=False,
                   fig_path="./figures",
                   fig_name="range01 val loss by batch seq")

plot_loss_by_batch(loss_path=["./checkpoints/unet_1_range01",
                              "./checkpoints/unet_1_range01"],
                   epoch_counts=[2, 2],
                   ma_k_size=5000,
                   dataset="train",
                   fig_save=False,
                   fig_path="./figures",
                   fig_name="range01 train loss by batch")

plot_loss_by_epoch(loss_path=["./checkpoints/unet_1_range01",
                              "./checkpoints/unet_1_range01"],
                   fig_save=False,
                   fig_path="./figures",
                   fig_name="range01 train+val loss per epoch")


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
