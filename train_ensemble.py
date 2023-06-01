import torch
import wandb
import hydra
import os
import clip
import torch.optim as optim
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from augments.augmentationtransforms import AugmentationTransforms
from models.ensemble_model import EnsembleModel
from models.ensemble_model_list import EnsembleModelList

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    # os.environ['WANDB_API_KEY'] = '045006204280bf2b17bd53dfd35a0ba8e54d00b6'
    # os.environ['WANDB_MODE'] = 'offline'

    wandbname = 'Ensemble_4models'
    learning_rate = 5e-8
    wd = 0.01
    betas = (0.9, 0.999)
    activate_checkpoint = False

    logger = wandb.init(project="report tests", name=wandbname)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    name_changer = {'entoloma lividum' : 'an entoloma lividium mushroom',
                    'salvelinus fontinalis' : 'a salvelinus fontinalis fish',
                    'bearberry' : 'a red bearberry fruit',
                    'brick red' : 'a red brick house or landscape',
                    'carbine' : 'a carbine pistol weapon',
                    'ceriman' : 'a green ceriman fruit or landscape',
                    'couscous' : 'an oriental couscous',
                    'flash' : 'rainbow flash room',
                    'florist' : 'florist flowers',
                    'kingfish' : 'a kingfish fish',
                    'organ loft' : 'church organ loft',
                    'peahen' : 'a peahen bird',
                    'plunge' : 'pool water plunge',
                    'silkworm' : 'a worm',
                    'veloute' : 'a veloute soup in a cup',
                    'vintage' : 'a vintage building or castle',
                    'zinfandel' : 'red wine glass or bottle'}
    name_changerV2 = {
        'bat': 'a bat',
        'bearberry' : 'a red bearberry fruit',
        'black tailed deer' : 'a deer',
        'brick red' : 'a red brick house or landscape',
        'carbine' : 'a carbine rifle pistol weapon',
        'ceriman' : 'a green ceriman fruit or landscape',
        'couscous' : 'an oriental granular couscous',
        'entoloma lividum' : 'an entoloma lividium brown mushroom',
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

    name_changerV3 = {
        'bat': 'a bat',
        'black tailed deer' : 'a deer',
        'carbine' : 'a carbine rifle pistol weapon',
        'couscous' : 'an oriental granular couscous',
        'ethyl alcohol' : 'alcohol effects',
        'florist' : 'florist flowers',
        'gosling' : 'a gosling or Ryan Gosling',
        'grenadine' : 'a grenade red fruity mood picture',
        'organ loft' : 'a church organ loft with stainglass',
        'platter' : 'a platter plate',
        'zinfandel' : 'red wine glass bottle or grape field'}
    
    class_to_idx = datamodule.dataset.class_to_idx

    class_list0 = list(range(48))
    class_list1 = list(range(48))
    class_list2 = list(range(48))
    class_list3 = list(range(48))

# For ensemble learning

    for  (class_name, index) in class_to_idx.items():
        class_name = class_name.lower()
        class_list0[index] = class_name
        if class_name in name_changer.keys():
            class_list1[index] = name_changer[class_name]
        else:
            class_list1[index] = class_name
        if class_name in name_changerV2.keys():
            class_list2[index] = name_changerV2[class_name]
        else:
            class_list2[index] = class_name
        if class_name in name_changerV3.keys():
            class_list3[index] = name_changerV3[class_name]
        else:
            class_list3[index] = class_name

    text0= clip.tokenize(class_list0).to(device)
    text1= clip.tokenize(class_list1).to(device)
    text2= clip.tokenize(class_list2).to(device)
    text3= clip.tokenize(class_list3).to(device)

    # Creation and loading the model 0
    model0, preprocess0 = clip.load("ViT-B/16", device=device)
    checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    path = os.path.join(checkpoints_path, 'REPORT_clip16_lr5e-8_wd.01_simple_allunfroz_chckpt_final.pt')
    checkpoint = torch.load(path)
    model0.load_state_dict(checkpoint)
    model0.float()

    # Creation and loading the model 1
    model1, preprocess1 = clip.load("ViT-B/16", device=device)
    checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    path = os.path.join(checkpoints_path, 'REPORT_clip16_lr5e-8_wd.01_simple_allunfroz_namechg_chckpt_final.pt')
    checkpoint = torch.load(path)
    model1.load_state_dict(checkpoint)
    model1.float()

    # Creation and loading the model 2
    model2, preprocess2 = clip.load("ViT-B/16", device=device)
    checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    path = os.path.join(checkpoints_path, 'REPORT_clip16_lr5e-8_wd.01_simple_allunfroz_namechgV2_chckpt_30.pt')
    checkpoint = torch.load(path)
    model2.load_state_dict(checkpoint)
    model2.float()

    # Creation and loading the model 3
    model3, preprocess3 = clip.load("ViT-B/16", device=device)
    checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    path = os.path.join(checkpoints_path, 'REPORT_clip16_lr5e-8_wd.01_simple_allunfroz_namechgV3_chckpt_25.pt')
    checkpoint = torch.load(path)
    model3.load_state_dict(checkpoint)
    model3.float()

    

    models = [model0, model1, model2, model3]
    texts = [text0, text1, text2, text3]

    model = EnsembleModelList(models, texts, cfg.dataset.num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd, betas=betas)
    # optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())

    # train_dataset = datamodule.train_dataset
    # val_loader = datamodule.val_dataloader()

    simple_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224),antialias=None),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    train_path = 'train'
    traindir = os.path.join(cfg.data_dir, train_path)
    valdir = os.path.join(cfg.data_dir, 'val_train')

    train_dataset = ImageFolder(traindir, transform=simple_transform)
    train_loader = DataLoader(train_dataset, batch_size=datamodule.batch_size, shuffle=True, num_workers=datamodule.num_workers)
    val_dataset = ImageFolder(valdir, transform=simple_transform)
    val_loader = DataLoader(val_dataset, batch_size=datamodule.batch_size, shuffle=False, num_workers=datamodule.num_workers)

    # train_loader = datamodule.train_dataloader()
    # val_loader = datamodule.val_dataloader()

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        for _, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
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

        if epoch%10 == 0 and activate_checkpoint :
            checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
            torch.save(model.state_dict(), os.path.join(checkpoints_path, f'{wandbname}_chckpt_{epoch}.pt'))

        for _, batch in enumerate(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
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