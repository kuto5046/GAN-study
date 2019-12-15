import os
import random
import math
import time
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from model import Self_Attention, Generator, Discriminator, weights_init
from train import train_model
from data_pipeline import load_dataset, make_datapath_list, ImageTransform, GAN_Img_Dataset

# Setup seeds
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything()


# 損失の可視化
def loss_process(G_losses, D_losses):
    loss_path = "./output/loss/"
    os.makedirs(loss_path, exist_ok=True)
    sns.set_style("white")
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("G and D Loss During Training")
    ax.plot(G_losses,label="G",c="b")
    ax.plot(D_losses,label="D",c="r")
    ax.set_xlabel("iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(loss_path + "loss.jpg")
    plt.show()


def main():
    # 使用パラメータ
    nz = 100
    mini_batch_size = 64
    image_size = 64
    num_epochs = 2

    # GPUの選択
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)  

    # MNISTデータの読み込み
    load_dataset()

    # Datasetを作成
    train_data_list = make_datapath_list()
    mean = (0.5, )
    std = (0.5, )
    train_dataset = GAN_Img_Dataset(
        file_list = train_data_list, transform=ImageTransform(mean, std))

    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    
    # インスタンス変数を作成
    G = Generator(nz=nz, image_size=image_size)
    D = Discriminator(nz=nz, image_size=image_size)

    # 初期化の実施
    G.apply(weights_init)
    D.apply(weights_init)
    print("Finish initialize network")

    # 訓練開始
    G_update, D_update, G_losses, D_losses = train_model(G, D, train_dataloader, num_epochs, nz, mini_batch_size, device)

    # 損失関数の可視化
    loss_process(G_losses, D_losses)

if __name__ == "__main__":
    main()
 