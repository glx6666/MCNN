import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.io import loadmat
from PIL import Image
from torchvision import transforms as transforms
import os

class CustomCWRUDataset(Dataset):
    def __init__(self, data_pd):
        super(CustomCWRUDataset, self).__init__()
        self.data_pd = data_pd
        self.transforms = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
        file_path = self.data_pd.iloc[idx]['data']
        label = self.data_pd.iloc[idx]['label']
        #data = loadmat(file_path)['sample'].transpose()
        #data = torch.tensor(data).float()
        img_data = Image.open(file_path)
        img_data = img_data.convert('RGB')
        label = int(torch.tensor(label))
        img_data = self.transforms(img_data)
        return img_data, label

