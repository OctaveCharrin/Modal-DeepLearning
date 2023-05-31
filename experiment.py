import torch
# import wandb
import hydra
# import os
# from tqdm import tqdm
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
# from torchvision.datasets import DatasetFolder
# from augments.augmentationtransforms import AugmentationTransforms
from data.datamodule import DataModule


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    class_to_idx = datamodule.dataset.class_to_idx

    name_changerV2 = {
        'bat': 'a bat',
        'bearberry' : 'a red bearberry fruit',
        'black tailed deer' : 'a deer',
        'brick red' : 'a red brick house or landscape',
        'carbine' : 'a carbine rifle pistol weapon',
        'ceriman' : 'a green ceriman fruit or landscape',
        'couscous' : 'an oriental granular couscous',
        'entoloma lividum' : 'an entoloma lividium mushroom',
        'ethyl alcohol' : 'alcohol effects',
        'flash' : 'rainbow flash room',
        'florist' : 'florist flowers',
        'gosling' : 'a gosling or Ryan Gosling',
        'grenadine' : 'a grenade red fruity mood picture',
        'kingfish' : 'a kingfish fish',
        'organ loft' : 'a church organ loft with stainglass',
        'peahen' : 'a peahen bird',
        'platter' : 'a platter plate',
        'plunge' : 'pool water plunge',
        'salvelinus fontinalis' : 'a salvelinus fontinalis fish',
        'silkworm' : 'a worm',
        'veloute' : 'a veloute soup in a cup',
        'vintage' : 'a vintage building or castle',
        'zinfandel' : 'red wine glass bottle or grape field'}

    
    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_name = class_name.lower()
        if class_name in name_changerV2.keys():
            class_list[index] = name_changerV2[class_name]
        else :
            class_list[index] = class_name


    m=  ['cavern', 'red', 'lavender', 'building', 'dragon', 'owl', 'cherry', 
    'mountain', 'colorful', 'screens', 'sand', 'purple', 'subway', 'pilar', 
    'psychadelic', 'castle', 'grass', 'baloon', 'pipe', 'space', 'forest']
    print(len(list(set(class_list+m))))



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