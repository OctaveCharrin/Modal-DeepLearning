import torch
import torch.nn as nn
import clip
import os

class EnsembleModelList(nn.Module):
    def __init__(self, models, texts, num_classes):
        super(EnsembleModelList, self).__init__()

        assert len(models) != 0
        assert len(models) == len(texts)
        self.num_models = len(models)

        # Freeze the parameters of the backbone models
        for model in models:
            for param in model.parameters():
                param.requires_grad = False

        # Store the backbone models
        self.models = models
        self.texts = texts

        # Learnable fusion layer
        self.fusion_layer = nn.Linear(in_features=num_classes * len(models), out_features=num_classes)
        
        # Initialize the fusion layer weights and biases for sum operation
        
        with torch.no_grad():
            stacked_matrix=torch.eye(num_classes)
            for _ in range(len(models)-1):
                eye_matrix = torch.eye(num_classes)
                stacked_matrix = torch.cat((stacked_matrix, eye_matrix), dim=1)

            self.fusion_layer.weight.copy_(stacked_matrix)
            self.fusion_layer.bias.copy_(torch.zeros(num_classes))

    def forward(self, x):
        concatenated, _ = self.models[0](x, self.texts[0])
        for i in range(1, self.num_models):
            output, _ = self.models[i](x, self.texts[i])
            concatenated = torch.cat((concatenated, output), dim=1)

        # Pass the concatenated output through the fusion layer
        fused_output = self.fusion_layer(concatenated)

        return fused_output
    
    def create_model(cfg, device, datamodule):
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
            return model