import os

from keras.utils import np_utils
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


def load_data(path):
    """
    Load CIFAR10 data
    Reference:
      https://www.kaggle.com/vassiliskrikonis/cifar-10-analysis-with-a-neural-network/data

    """
    def _load_batch_file(batch_filename):
        filepath = os.path.join(path, batch_filename)
        unpickled = _unpickle(filepath)
        return unpickled

    def _unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin')
        return dict

    train_batch_1 = _load_batch_file('data_batch_1')
    train_batch_2 = _load_batch_file('data_batch_2')
    train_batch_3 = _load_batch_file('data_batch_3')
    train_batch_4 = _load_batch_file('data_batch_4')
    train_batch_5 = _load_batch_file('data_batch_5')
    test_batch = _load_batch_file('test_batch')

    num_classes = 10
    batches = [train_batch_1['data'], train_batch_2['data'], train_batch_3['data'], train_batch_4['data'], train_batch_5['data']]
    train_x = np.concatenate(batches)
    train_x = train_x.astype('float32') # this is necessary for the division below

    train_y = np.concatenate([np_utils.to_categorical(labels, num_classes) for labels in [train_batch_1['labels'], train_batch_2['labels'], train_batch_3['labels'], train_batch_4['labels'], train_batch_5['labels']]])
    test_x = test_batch['data'].astype('float32') #/ 255
    test_y = np_utils.to_categorical(test_batch['labels'], num_classes)
    print(num_classes)
   
    img_rows, img_cols = 32, 32
    channels = 3
    print(train_x.shape)
    train_x = train_x.reshape(len(train_x), channels, img_rows, img_cols)
    test_x = test_x.reshape(len(test_x), channels, img_rows, img_cols)
    train_x = train_x.transpose((0, 2, 3, 1))
    test_x = test_x.transpose((0, 2, 3, 1))
    per_pixel_mean = (train_x).mean(0) # 計算はするが使用しない

    train_x = [Image.fromarray(img.astype(np.uint8)) for img in train_x]
    test_x = [Image.fromarray(img.astype(np.uint8)) for img in test_x]
    
    train = [(x,np.argmax(y)) for x, y in zip(train_x, train_y)]
    test = [(x,np.argmax(y)) for x, y in zip(test_x, test_y)]
    return train, test, per_pixel_mean


class ImageDataset(Dataset):
    """
    データにtransformsを適用するためのクラス
    """
    def __init__(self, data, transform=None):

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        
        if self.transform:
            img = self.transform(img)

        return img, label


# Googleドライブのマウント
from google.colab import drive
drive.mount('./drive')

BATCH_SIZE = 128
# path = "./drive/My Drive/Colab Notebooks/dataset/cifar-10-batches-py/"
path = ""
train, test = load_data(path)


# train dataの作成
train_transform = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda img: np.array(img)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.float()),
])
train_dataset = ImageDataset(train[:45000], transform=train_transform)
trainloader = DataLoader(train_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=0)


# validation data, test dataの作成
valtest_transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda img: np.array(img)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.float()),    
    ])
valid_dataset = ImageDataset(train[45000:], transform=valtest_transform)
validloader = DataLoader(valid_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=0)

test_dataset = ImageDataset(test, transform=valtest_transform)
testloader = DataLoader(test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)