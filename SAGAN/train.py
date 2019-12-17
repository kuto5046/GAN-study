import os
import pickle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim


# 生成画像を可視化する
def generate_img(G, epoch, fixed_z, device):
    
    generate_img_path = "./output/generate_img/"
    os.makedirs(generate_img_path, exist_ok=True)
    
    # 画像生成
    fake_images, attention_map1, attention_map2 = G(fixed_z.to(device))

    # 出力
    fig = plt.figure(figsize=(15, 6))
    plt.title("epoch = {}".format(epoch))
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                    bottom=False,
                    left=False,
                    right=False,
                    top=False)

    for i in range(5):
        ax = fig.add_subplot(3, 5, i+1)
        ax.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
        ax.set_axis_off()

        ax = fig.add_subplot(3, 5, i+6)
        ax.imshow(attention_map1[i].cpu().detach().numpy(), 'gray')
        ax.set_axis_off()

        ax = fig.add_subplot(3, 5, i+11)
        ax.imshow(attention_map2[i].cpu().detach().numpy(), 'gray')
        ax.set_axis_off()

    
    plt.savefig(generate_img_path + "Generate_epoch{}.jpg".format(epoch))


def train_model(G, D, dataloader, num_epochs, nz, mini_batch_size, device):
    # 最適化手法の設定
    G_lr, D_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    optimizerG = torch.optim.Adam(G.parameters(), G_lr, [beta1, beta2]) 
    optimizerD = torch.optim.Adam(D.parameters(), D_lr, [beta1, beta2])
    
    # 誤差関数の定義
    # criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # ネットワークをGPUへ
    G.to(device)
    D.to(device)

    # 訓練モードへ切り替え
    G.train()
    D.train()

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # 画像生成可視化用
    fixed_z = torch.randn(5, nz)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    iteration = 1

    # 各epochでの損失を記録
    G_losses = []
    D_losses = []

    print("Start training!")
    for epoch in tqdm(range(num_epochs)):

        # start_time = time.time()

        # print("Epoch {}/{}".format(epoch, num_epochs))

        for data in dataloader:

            # --------------------
            # 1. Update D network
            # --------------------
            # ミニバッチが１だとBatchNormでエラーが出るので回避
            if data.size()[0] == 1:
                continue

            # GPU使えるならGPUにデータを送る
            data = data.to(device)
            
            # ラベルの作成
            mini_batch_size = data.size()[0]
            # real_label = torch.full((mini_batch_size,), 1).to(device)
            # fake_label = torch.full((mini_batch_size,), 0).to(device)

            # 真の画像を判定
            D_real_output, _, _ = D(data)
            
            # 偽画像を生成して判定
            z = torch.randn(mini_batch_size, nz).to(device)
            z = z.view(z.size(0), z.size(1), 1, 1)
            fake_imgs, _, _ = G(z)
            D_fake_output, _, _ = D(fake_imgs)

            # 誤差を計算 (hinge version of the adversarial lossに変更)
            """
            ・lossD_real
            誤差d_out_realが1以上で誤差0になる
            D_real_output>1で, (1.0-D_fake_output)が負の場合ReLUで0になる

            ・lossD_real
            誤差d_out_fakeが-1以下なら誤差0になる
            D_fake_output<-1で, (1.0+D_real_output)が負の場合ReLUで0になる  
            """
            lossD_real = torch.nn.ReLU()(1.0 - D_real_output).mean()
            lossD_fake = torch.nn.ReLU()(1.0 + D_fake_output).mean()
            lossD = lossD_real + lossD_fake

            # 誤差逆伝播
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()

            # --------------------
            # 2. Update G network
            # --------------------
            # 偽画像を生成して判定
            z = torch.randn(mini_batch_size, nz).to(device)
            z = z.view(z.size(0), z.size(1), 1, 1)
            fake_imgs, _, _ = G(z)
            D_fake_output, _, _ = D(fake_imgs)

            # 誤差を計算
            lossG = - D_fake_output.mean()

            # 誤差逆伝播
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            lossG.backward()
            optimizerG.step()

            # -----------
            # 3. 記録
            # -----------
            D_losses.append(lossD.item())
            G_losses.append(lossG.item())
            iteration += 1
        
        # 画像生成
        if epoch % 20 == 0: 
            generate_img(G, epoch, fixed_z, device)
        
    # モデルの保存
    os.makedirs("./output/model/", exist_ok=True)
    with open("./output/model/G_model.pickle", mode="wb") as fp:
        pickle.dump(G, fp)
    with open("./output/model/D_model.pickle", mode="wb") as fp:
        pickle.dump(D, fp)
        
    return G, D, G_losses, D_losses