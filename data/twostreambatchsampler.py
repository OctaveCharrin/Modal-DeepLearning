import torch
from torch.utils.data import Sampler



class TwoStreamBatchSampler(Sampler):
    def __init__(self, main_dataset, augmented_dataset, main_batch_size, augmented_batch_size):
        self.main_dataset = main_dataset
        self.augmented_dataset = augmented_dataset
        self.main_batch_size = main_batch_size
        self.augmented_batch_size = augmented_batch_size

        self.main_sampler = torch.utils.data.RandomSampler(self.main_dataset)
        self.augmented_sampler = torch.utils.data.RandomSampler(self.augmented_dataset)

    def __iter__(self):
        main_batch = iter(torch.utils.data.BatchSampler(self.main_sampler, self.main_batch_size, drop_last=True))
        augmented_batch = iter(torch.utils.data.BatchSampler(self.augmented_sampler, self.augmented_batch_size, drop_last=True))

        for _ in range(len(self)):
            yield [next(main_batch) for _ in range(self.main_batch_size)] + [next(augmented_batch) for _ in range(self.augmented_batch_size)]

    def __len__(self):
        return min(len(self.main_dataset) // self.main_batch_size, len(self.augmented_dataset) // self.augmented_batch_size)