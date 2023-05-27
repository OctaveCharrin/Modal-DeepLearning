import torch
import wandb
import hydra
import os
import clip
import torch.optim as optim
import numpy as np
# from timm.data.auto_augment import rand_augment_transform
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from augments.augmentationtransforms import AugmentationTransforms
from augments.addgaussiannoise import AddGaussianNoise


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    # os.environ['WANDB_API_KEY'] = '045006204280bf2b17bd53dfd35a0ba8e54d00b6'
    # os.environ['WANDB_MODE'] = 'offline'

    wandbname = 'run_clip16_wd10_simple_allmlpunfrozen'
    learning_rate = 1e-7
    wd = 10
    aug_num = 3
    final = False
    resume = False
    numcheck = 5

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    augments = AugmentationTransforms().toList()

    # augments = [
    #     transforms.Compose([transforms.RandomCrop((180,180)),transforms.Resize((224,224))]),
    #     transforms.Compose([transforms.RandomCrop((300,300), pad_if_needed=True),transforms.Resize((224,224))]),
    #     transforms.Compose(5*[transforms.RandomErasing(p=.75, scale=(0.01, 0.05), ratio=(0.5, 1.5))]),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.Compose([AddGaussianNoise(mean=0.0, std=0.2),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    #     transforms.Compose([transforms.RandomRotation(degrees=180,expand=True),transforms.Resize((224,224))])
    #     ]

    random_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224),antialias=None),
                transforms.RandomHorizontalFlip(p=.5),
                transforms.RandomVerticalFlip(p=.5),
                AddGaussianNoise(mean=0.0, std=0.2),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    simple_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224),antialias=None),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    

    train_path = 'train' if final else 'train_val'
    traindir = os.path.join(cfg.data_dir, train_path)
    valdir = os.path.join(cfg.data_dir, 'val_train')

    train_dataset = ImageFolder(traindir, transform=simple_transform)
    # train_dataset = ImageFolder(traindir, transform=random_transform)

    val_dataset = ImageFolder(valdir, transform=simple_transform)
    class_to_idx = train_dataset.class_to_idx

    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_list[index] = class_name

    
    logger = wandb.init(project="challenge", name=wandbname)
    
    val_loader = DataLoader(val_dataset, batch_size=datamodule.batch_size, shuffle=False, num_workers=datamodule.num_workers)

    # model, preprocess = clip.load("ViT-B/32", device=device)
    model, preprocess = clip.load("ViT-B/16", device=device)

    if resume :
        checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
        path = os.path.join(checkpoints_path, 'heckpoint_final.pt')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)

    model.float()

    for name, param in model.named_parameters():
        # if ('mlp' in name) and (name.startswith('visual')):
        if ('mlp' in name):
            continue
        else:
            param.requires_grad = False

    text = clip.tokenize(class_list).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    # optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())

    train_loader = DataLoader(train_dataset, batch_size=datamodule.batch_size, shuffle=True, num_workers=datamodule.num_workers)

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        # sampled_ops = np.random.choice(augments, aug_num)
        # sampled_aug = transforms.Compose(sampled_ops)
        # train_augment = transforms.Compose([simple_transform,sampled_aug])
        # train_dataset.transform = train_augment
        # train_loader = DataLoader(train_dataset, batch_size=datamodule.batch_size, shuffle=True, num_workers=datamodule.num_workers)

        for _, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            logits_per_image, logits_per_text = model(images, text)

            loss = loss_fn(logits_per_image, labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (logits_per_image.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        if epoch%numcheck == 0 and not final:
            checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
            torch.save(model.state_dict(), os.path.join(checkpoints_path, f'{wandbname}_chckpt_{epoch}.pt'))

        for _, batch in enumerate(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits_per_image, logits_per_text = model(images, text)
            loss = loss_fn(logits_per_image, labels)
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (logits_per_image.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)

        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "val_loss_epoch": epoch_loss,
                "val_acc": epoch_acc,
            }
        )
    checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    torch.save(model.state_dict(), os.path.join(checkpoints_path, f'{wandbname}_chckpt_final.pt'))
    wandb.finish()


if __name__ == "__main__":
    train()