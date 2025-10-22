import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ulit.cwru_datasets import *


class CWRU(object):
    def __init__(self, root_dir, test_size=0.2, transform=None):
        self.root_dir = root_dir
        self.test_size = test_size
        self.data_pd = self.load_cwru_data()
        self.transform = transform

    def load_cwru_data(self):
        data = {'data': [], 'label': []}
        for class_label, class_name in enumerate(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    data['data'].append(file_path)
                    data['label'].append(class_label)
        return pd.DataFrame(data)

    def train_test_split_order(self):
        train_pd, test_pd, _, _ = train_test_split(
            self.data_pd,
            self.data_pd['label'],
            test_size=self.test_size,
            stratify=self.data_pd['label'],  # Ensure stratified split based on labels
            random_state=123  # Set a seed for reproducibility
        )
        train_dataset = CustomCWRUDataset(train_pd)
        test_dataset = CustomCWRUDataset(test_pd)
        return train_dataset, test_dataset
