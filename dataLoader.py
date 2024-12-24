import os
from torch.utils.data import Dataset
import pandas as pd
import rasterio
import torch
from utils.utility import sort_filenames_by_number
class CustomDataset(Dataset):
    def __init__(self, data_dir,output=None, transform=None):
        self.data = []
        self.root_dir = data_dir
        self.transform = transform
        self.files = os.listdir(self.root_dir)
        self.files = sort_filenames_by_number(self.files)
        if output:
            self.output = pd.read_csv(output,header=None)
            mapper = dict(zip(list(self.output[0]),list(self.output[2])))
            for i in self.files:
                self.data.append((os.path.join(self.root_dir,i),mapper[i]))
        else:
            for i in self.files:
                self.data.append((os.path.join(self.root_dir,i),i))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path, label_index = self.data[idx]
        
        # Load image file
        with rasterio.open(file_path) as fl:
            image = fl.read()
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        if not isinstance(label_index, str):
            label_index = torch.tensor(label_index)

        
        return torch.tensor(image), label_index