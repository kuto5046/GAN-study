import os
import urllib.request
import zipfile
import tarfile
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from sklearn.datasets import fetch_openml


def load_dataset():
    # フォルダ「data」が存在しない場合は作成する
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    mnist = fetch_openml('mnist_784', version=1, data_home=data_dir)  

    # データの取り出し
    X = mnist.data
    y = mnist.target

    # フォルダ「data」の下にフォルダ「img_78」を作成する
    data_dir_path = "./data/img_01234/"
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    # MNISTから数字7、8の画像だけフォルダ「img_78」に画像として保存していく
    count0=0
    count1=0
    count2=0
    max_num=200  # 画像は200枚ずつ作成する
        
    for i in range(len(X)):                       
        # 画像0の作成
        if (y[i] is "0") and (count0<max_num):
            file_path="./data/img_01234/img_0_"+str(count0)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count0+=1 
        
        # 画像1の作成
        if (y[i] is "1") and (count1<max_num):
            file_path="./data/img_01234/img_1_"+str(count1)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count1+=1 

        # 画像2の作成
        if (y[i] is "2") and (count2<max_num):
            file_path="./data/img_01234/img_2_"+str(count2)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count2+=1 
    print("loaded data")


def make_datapath_list():
    train_img_list = []

    for img_idx in range(200):

        img_path = "./data/img_01234/img_0_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

        img_path = "./data/img_01234/img_1_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

        img_path = "./data/img_01234/img_2_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

        # img_path = "./data/img_01234/img_3_" + str(img_idx) + ".jpg"
        # train_img_list.append(img_path)

        # img_path = "./data/img_01234/img_4_" + str(img_idx) + ".jpg"
        # train_img_list.append(img_path)

    return train_img_list

class ImageTransform():

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        """
        Pythonのメソッド 
        そのクラスのインスタンスが具体的な関数を指定されずに
        呼び出された時に動作する関数
        """
        return self.data_transform(img)


class GAN_Img_Dataset(data.Dataset):
    """
    画像のDatasetクラス
    PyTorchのDatasetクラスを継承
    Datasetクラス継承する際には__len__()と__getitem__()を実装する必要がある
    """
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        # 画像の枚数を返す
        return len(self.file_list)

    def __getitem__(self, index):
        # 前処理した画像のTensor形式のデータを取得
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        return img_transformed