from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
import hydra
import torch
from torchvision import transforms

@hydra.main(config_path="configs", config_name="config", version_base=None)
def experiment(cfg):

    unlabelled_dataset = ImageFolder(cfg.datasetmodule.test_dataset_path, transform=transforms.ToTensor())
    res=[]
    for i in range(10):
        for j in range(100):
            item = unlabelled_dataset[i*6000+j]
            item = (item[0],i)
            res.append(item)

    test_dataset = torch.tensor(res)
    
    return test_dataset


if __name__ == "__main__":
    experiment()