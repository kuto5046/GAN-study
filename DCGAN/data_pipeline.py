from PIL import Image
import torch.utils.data as data
from torchvision import transforms


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