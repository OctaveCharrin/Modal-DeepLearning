from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
import torch


class DatasetModule:
    def __init__(
        self,
        train_dataset_path,
        train_transform,
        test_dataset_path,
        test_transform,
        batch_size,
        num_workers,
    ):
        self.train_dataset = ImageFolder(train_dataset_path, transform=train_transform)
        self.test_dataset = ImageFolder(test_dataset_path, transform=test_transform)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
