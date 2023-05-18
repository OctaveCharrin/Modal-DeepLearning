from torchvision import transforms
from augments.addgaussiannoise import AddGaussianNoise

class AugmentationTransforms:
    def __init__(
        self,
    ):
        
        self.simple = transforms.Compose([
                                         transforms.Resize((240,240)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
        self.random_crop_small = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.Resize((224,224)),
                                                    transforms.RandomCrop((180,180)),
                                                    transforms.Resize((224,224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                    ])
        self.random_crop_big = transforms.Compose([
                                                  transforms.ToPILImage(),
                                                  transforms.Resize((224,224)),
                                                  transforms.RandomCrop((300,300), pad_if_needed=True),
                                                  transforms.Resize((224,224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                  ])
        self.random_erasing = transforms.Compose([
                                                 transforms.ToPILImage(),
                                                 transforms.Resize((240,240)),
                                                 transforms.ToTensor(),
                                                 transforms.RandomErasing(p=.75, scale=(0.01, 0.05), ratio=(0.5, 1.5)),
                                                 transforms.RandomErasing(p=.75, scale=(0.01, 0.05), ratio=(0.5, 1.5)),
                                                 transforms.RandomErasing(p=.75, scale=(0.01, 0.05), ratio=(0.5, 1.5)),
                                                 transforms.RandomErasing(p=.75, scale=(0.01, 0.05), ratio=(0.5, 1.5)),
                                                 transforms.RandomErasing(p=.75, scale=(0.01, 0.05), ratio=(0.5, 1.5)),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                 ])
        self.horizontal_flip = transforms.Compose([
                                                  transforms.ToPILImage(),
                                                  transforms.Resize((240,240)),
                                                  transforms.RandomHorizontalFlip(p=1.),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                  ])
        self.vertical_flip = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.Resize((240,240)),
                                                transforms.RandomVerticalFlip(p=1.),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])
        self.random_gaussian_noise = transforms.Compose([
                                                        transforms.ToPILImage(),
                                                        transforms.Resize((224,224)),
                                                        transforms.ToTensor(),
                                                        AddGaussianNoise(mean=0.0, std=0.2),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                        ])
        self.random_rotation = transforms.Compose([
                                                  transforms.ToPILImage(),
                                                  transforms.Resize((224,224)),
                                                  transforms.RandomRotation(degrees=180,expand=True),
                                                  transforms.Resize((224,224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                  ])


    def toList(self):
        augmentList = [self.simple,
                       self.random_crop_small,
                       self.random_crop_big,
                       self.random_erasing,
                       self.horizontal_flip,
                       self.vertical_flip,
                       self.random_rotation,
                       self.random_gaussian_noise,
                       ]
        return augmentList