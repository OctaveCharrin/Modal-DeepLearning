import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from include import architectures
from include.utils import *


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

    checkpoint_path = 'CHECKPOINT_PATH.ckpt'

    cwd = os.getcwd()

    # Use the correct path for datasets
    test_path = os.path.join(cwd, '../dataset/test')
    class_path = os.path.join(cwd, '../dataset/train')

    class_names = sorted(os.listdir(class_path))

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

    # Create model and load state dict from checkpoint_path
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
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, image_names = batch
            images = images.to(device)
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
