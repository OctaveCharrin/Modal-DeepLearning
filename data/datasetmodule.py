from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch

import os
from PIL import Image


class UnlabelledDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.folder_path, self.file_list[index])
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, index



class DatasetModule:
    def __init__(
        self,
        train_dataset_path,
        train_transform,
        unlabelled_dataset_path,
        unlabelled_transform,
        batch_size,
        num_workers,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = ImageFolder(train_dataset_path, transform=train_transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [
                int(0.8 * len(self.dataset)),
                len(self.dataset) - int(0.8 * len(self.dataset)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )
        self.unlabelled_dataset = UnlabelledDataset(unlabelled_dataset_path, transform=unlabelled_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def unlabelled_dataloader(self):
        return DataLoader(
            self.unlabelled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )