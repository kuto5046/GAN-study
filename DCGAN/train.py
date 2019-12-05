import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim


# 生成画像と訓練データを可視化する
def generate_img(G, train_dataloader, epoch, batch_size, nz, device):
    generate_img_path = "./output/generate_img/"
    os.makedirs(generate_img_path, exist_ok=True)

    fixed_z = torch.randn(batch_size, nz)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    # 画像生成
    fake_images = G(fixed_z.to(device))

    # 訓練データ
    batch_iterator = iter(train_dataloader)  # イテレータに変換
    imges = next(batch_iterator)  # 1番目の要素を取り出す


    # 出力
    fig = plt.figure(figsize=(15, 6))
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                    bottom=False,
                    left=False,
                    right=False,
                    top=False)
    
    for i in range(0, 5):
        # 上段に訓練データを
        plt.subplot(2, 5, i+1)
        plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')

        # 下段に生成データを表示する
        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

    plt.savefig(generate_img_path + "Generate_epoch{}.jpg".format(epoch))


def train_model(G, D, dataloader, num_epochs, nz, mini_batch_size, device):
    # 最適化手法の設定
    G_lr, D_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    optimizerG = torch.optim.Adam(G.parameters(), G_lr, [beta1, beta2]) 
    optimizerD = torch.optim.Adam(D.parameters(), D_lr, [beta1, beta2])

    # 誤差関数の定義
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # ネットワークをGPUへ
    G.to(device)
    D.to(device)

    # 訓練モードへ切り替え
    G.train()
    D.train()

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    print("Start training!")
    for epoch in tqdm(range(num_epochs)):

        # start_time = time.time()

        # 各epochでの損失を記録
        G_losses = []
        D_losses = []

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
            real_label = torch.full((mini_batch_size,), 1).to(device)
            fake_label = torch.full((mini_batch_size,), 0).to(device)

            # 真の画像を判定
            D_real_output = D(data)
            
            # 偽画像を生成して判定
            z = torch.randn(mini_batch_size, nz).to(device)
            z = z.view(z.size(0), z.size(1), 1, 1)
            fake_imgs = G(z)
            D_fake_output = D(fake_imgs)

            # 誤差を計算
            lossD_real = criterion(D_real_output.view(-1), real_label)
            lossD_fake = criterion(D_fake_output.view(-1), fake_label)
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
            fake_imgs = G(z)
            D_fake_output = D(fake_imgs)

            # 誤差を計算
            lossG = criterion(D_fake_output.view(-1), real_label)

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
            generate_img(G, dataloader, epoch, batch_size=8, nz=nz, device=device)
        # epochごとのphaseごとのlossと正解率
        # finish_time = time.time()
        # print('-------------')
        # print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
        #     epoch, D_losses/batch_size, G_losses/batch_size))
        # print('timer:  {:.4f} sec.'.format(finish_time - start_time))
        # start_time = time.time()
        
    return G, D, G_losses, D_losses