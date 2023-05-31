import torch
import wandb
import hydra
import os
import clip
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from augments.augmentationtransforms import AugmentationTransforms
from models.ensemble_model import EnsembleModel

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    val_dataloader = datamodule.val_dataloader()

    class_to_idx = datamodule.dataset.class_to_idx

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
    
    name_changer = name_changerV2
    
    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_name = class_name.lower()
        if class_name in name_changer.keys():
            class_list[index] = name_changer[class_name]
        else:
            class_list[index] = class_name
    
    text = clip.tokenize(class_list).to(device)

    zeroshot, _ = clip.load("ViT-B/16", device=device)
    finetuned, _ = clip.load("ViT-B/16", device=device)

    checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    path = os.path.join(checkpoints_path, 'FINAL_clip16_lr5e-8_wd.01_simple_allunfroz_namechg_chckpt_final.pt')
    checkpoint = torch.load(path)
    finetuned.load_state_dict(checkpoint)

    theta_0 = zeroshot.state_dict()
    theta_1 = finetuned.state_dict()

    best = (0,0)

    for alpha in np.linspace(0,1,20, endpoint=True):
        theta = {
            key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

        finetuned.load_state_dict(theta)
        acc = eval(finetuned, text, val_dataloader, device)

        print(f'accuracy for alpha = {alpha}: ', acc)
        if acc > best[0]:
            best = (acc, alpha)
    print(best)


def eval(model, text, dataloader, device):
    model.eval()

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            logits_per_image, logits_per_text = model(images, text)
            num_correct += (
                (logits_per_image.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        acc = num_correct / num_samples
    return acc


if __name__ == '__main__':
    main()