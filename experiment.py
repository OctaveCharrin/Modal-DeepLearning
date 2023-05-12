from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
import hydra
import torch
from torchvision import transforms
from augments.augmentationtransforms import AugmentationTransforms
import numpy as np
import matplotlib.pyplot as plt

@hydra.main(config_path="configs", config_name="config", version_base=None)
def experiment(cfg):
    t = AugmentationTransforms()
    augmentsList = t.toList()
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    dataset = datamodule.train_dataset
    

    for transform in augmentsList:
        plt.title(transform)
        img = dataset[0][0]
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()
        img = transform(img)
        npimg = img.numpy()
        plt.title(transform)
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()
        


if __name__ == "__main__":
    experiment()