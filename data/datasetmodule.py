from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


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
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = ImageFolder(train_dataset_path, transform=train_transform)

        self.unlabelled_dataset = ImageFolder(test_dataset_path, transform=test_transform)

        test_dataset=[]
        for i in range(10):
            for j in range(100):
                item = self.unlabelled_dataset[i*6200+j]
                item = (item[0],i)
                test_dataset.append(item)

        self.test_dataset = test_dataset

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
    
    def unlabelled_dataloader(self):
        return DataLoader(
            self.unlabelled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )