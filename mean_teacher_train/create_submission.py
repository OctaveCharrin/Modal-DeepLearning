from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch
from tqdm import tqdm
import clip
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision.transforms as transforms

from mean_teacher import architectures, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from torchvision.datasets import ImageFolder


import torch.nn as nn




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)

def create_submission():
    cwd = os.getcwd()
    test_path = os.path.join(cwd, '../datasetV2/test')
    train_path = os.path.join(cwd, '../datasetV2/train')


    simple_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224),antialias=None),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    test_loader = DataLoader(
        TestDataset(
            test_path, simple_transform
        ),
        batch_size=256,
        shuffle=False,
        num_workers=3,
    )

    # train_dataset = ImageFolder(train_path, transform=simple_transform)
    # class_to_idx = train_dataset.class_to_idx
    # class_list = list(range(48))
    # for  (class_name, index) in class_to_idx.items():
    #     class_name = class_name.lower()
    #     class_list[index] = class_name

    class_names = sorted(os.listdir(train_path))

    # Load model and checkpoint
    # model = hydra.utils.instantiate(cfg.model).to(device)
    # model, preprocess = clip.load("ViT-B/16", device=device)

    # checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    # path = os.path.join(checkpoints_path, 'RESUME_clip16_simple_finetune_mlpunfrozen_checkpoint_epoch_6.pt')
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint)

    # text = clip.tokenize(class_names).to(device)

    # model.eval()

    # For ensemble learning ###############

    # model1, preprocess = clip.load("ViT-B/16", device=device)
    # model2, preprocess = clip.load("ViT-B/16", device=device)

    # model1, model2 = model1.float(), model2.float()

    fname = 'Ensemble_4models_chckpt_final'

    # class_to_idx = datamodule.dataset.class_to_idx

    # name_changer = {'entoloma lividum' : 'an entoloma lividium mushroom',
    #                     'salvelinus fontinalis' : 'a salvelinus fontinalis fish',
    #                     'bearberry' : 'a red bearberry fruit',
    #                     'brick red' : 'a red brick house or landscape',
    #                     'carbine' : 'a carbine pistol weapon',
    #                     'ceriman' : 'a green ceriman fruit or landscape',
    #                     'couscous' : 'an oriental couscous',
    #                     'flash' : 'rainbow flash room',
    #                     'florist' : 'florist flowers',
    #                     'kingfish' : 'a kingfish fish',
    #                     'organ loft' : 'church organ loft',
    #                     'peahen' : 'a peahen bird',
    #                     'plunge' : 'pool water plunge',
    #                     'silkworm' : 'a worm',
    #                     'veloute' : 'a veloute soup in a cup',
    #                     'vintage' : 'a vintage building or castle',
    #                     'zinfandel' : 'red wine glass or bottle'}
                        
    # # name_changer = {}

    # class_list1 = list(range(48))
    # class_list2 = list(range(48))
    # for  (class_name, index) in class_to_idx.items():
    #     class_name = class_name.lower()
    #     class_list2[index] = class_name
    #     if class_name in name_changer.keys():
    #         class_list1[index] = name_changer[class_name]
    #     else:
    #         class_list1[index] = class_name

    
    # text1 = clip.tokenize(class_list1).to(device)
    # text2 = clip.tokenize(class_list2).to(device)

    # model = EnsembleModel(model1, model2, text1, text2, cfg.dataset.num_classes).to(device)

    # model = EnsembleModelList.create_model(cfg, device, datamodule)

    # checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    # path = os.path.join(checkpoints_path, f'{fname}.pt')
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint)
    # # Set the model to evaluation mode
    # model.eval()
    def create_model(ema=False):
        model_factory = architectures.__dict__['resnext152']
        model_params = dict(pretrained=False, num_classes=48)
        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    resume = '/home/octavecharrin/Desktop/Modal-DeepLearning/mean_teacher_train/results/main/2023-06-01_00-30-28/0/transient/checkpoint.1020.ckpt'
    
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])


    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, image_names = batch
            images = images.to(device)
            # preds, _ = model(images, text)
            preds, _ = model(images)
            preds = preds.argmax(1)
            preds = [class_names[pred] for pred in preds.cpu().numpy()]
            submission = pd.concat(
                [
                    submission,
                    pd.DataFrame({"id": image_names, "label": preds}),
                ]
            )
    submission.to_csv(os.path.join(cwd,"../submission.csv"), index=False)


if __name__ == "__main__":
    create_submission()
