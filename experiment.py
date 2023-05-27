import torch
# import wandb
# import hydra
# import os
# from tqdm import tqdm
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
# from torchvision.datasets import DatasetFolder
# from augments.augmentationtransforms import AugmentationTransforms


# @hydra.main(config_path="configs", config_name="config", version_base=None)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # unlabelled_transform = hydra.utils.instantiate(cfg.datamodule.train_transform)
    # unlabelled_dataset = UnlabelledDataset(cfg.datasetmodule.unlabeled_dataset_path, transform=unlabelled_transform)

    # dataloader = DataLoader(unlabelled_dataset, batch_size=8, shuffle = False)

    # datamodule = hydra.utils.instantiate(cfg.datamodule)
    # train_data = datamodule.train_dataset

    # val_loader = datamodule.val_dataloader()

    # model = hydra.utils.instantiate(cfg.model)

    # for i, batch in enumerate(val_loader):
    #     if i == 1:
    #         break
    #     images, labels = batch
    #     images.cuda()
    #     print("ook")
    #     pred = model(images)
    #     pred = torch.softmax(pred, dim=1)

    #     print('preds', pred)

    #     max_prob, predicted = torch.max(pred, dim=1)

    #     print('max', max_prob)
    #     print('mask', max_prob > 0.5)
    #     print('predicted', predicted)

    # data = TensorDataset(images, predicted)

    # print(data[0])

    # data = ConcatDataset([data,train_data])

    # new_dataloader = DataLoader(dataset=data, batch_size = 100, shuffle=True)
    # for i, batch in enumerate(new_dataloader):
    #     if i ==15:break
    #     print("ok")


    

        




# class UnlabelledDataset(Dataset):
#     def __init__(self, folder_path, transform=None):
#         self.folder_path = folder_path
#         self.file_list = os.listdir(folder_path)
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.file_list)
    
#     def __getitem__(self, index):
#         image_path = os.path.join(self.folder_path, self.file_list[index])
#         image = Image.open(image_path).convert("RGB")
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, -1
    


if __name__ == '__main__':
    main()