import torch.nn as nn
import clip
import os
import torch

CLASS_NAMES_1 = {
    'entoloma lividum' : 'an entoloma lividium mushroom',
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
    'zinfandel' : 'red wine glass or bottle'
    }

CLASS_NAMES_2 = {
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
    'zinfandel' : 'red wine glass bottle or grape field'
    }

CLASS_NAMES_3 = {
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
    'zinfandel' : 'red wine glass bottle or grape field'
    }

class ClipFinetune(nn.Module):
    def __init__(self, model, num_class, class_path, name_changer, device=torch.device('cpu')):
        super().__init__()
        assert model in ["ViT-B/16", "ViT-B/32"]

        class_names = sorted(os.listdir(class_path))
        mod, preprocess = clip.load(model, device=device)
        mod.float()
        self.model = mod
        if name_changer == 1:
            changer = CLASS_NAMES_1
        elif name_changer == 2:
            changer = CLASS_NAMES_2
        elif name_changer == 3:
            changer = CLASS_NAMES_3
        else :
            changer = {}

        self.class_names = list(range(num_class))
        for i, name in enumerate(class_names):
            name = name.lower()
            self.class_names[i] = name
            if name in changer:
                self.class_names[i] = changer[name]
        
        self.text = clip.tokenize(self.class_names).to(device)


    def forward(self, x):
        x, _ = self.model(x, self.text)
        return x