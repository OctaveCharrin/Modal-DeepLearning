from torchvision import transforms
from augments.addgaussiannoise import AddGaussianNoise

class AugmentationTransforms:
    def __init__(
        self, 
    ):
        self.color_jitter = transforms.Compose([
                                               transforms.ToPILImage(),
                                               transforms.Resize((240,240)),
                                               transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                               ])
        self.random_crop = transforms.Compose([
                                              transforms.ToPILImage(),
                                              transforms.Resize((224,224)),
                                              transforms.RandomCrop((180,180)),
                                              transforms.Resize((224,224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                              ])
        self.random_erasing = transforms.Compose([
                                                 transforms.ToPILImage(),
                                                 transforms.Resize((240,240)),
                                                 transforms.ToTensor(),
                                                 transforms.RandomErasing(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                 ])
        self.random_flip = transforms.Compose([
                                              transforms.ToPILImage(),
                                              transforms.Resize((240,240)),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                              ])
        self.random_gaussian_noise = transforms.Compose([
                                                        transforms.ToPILImage(),
                                                        transforms.Resize((224,224)),
                                                        AddGaussianNoise(mean=0.0, std=0.2),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                        ])
        self.random_rotation = transforms.Compose([
                                                  transforms.ToPILImage(),
                                                  transforms.Resize((224,224)),
                                                  transforms.RandomRotation(degrees=50,expand=True),
                                                  transforms.Resize((224,224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                  ])


    def toList(self):
        augmentList = [self.color_jitter,
                       self.random_crop,
                       self.random_erasing,
                       self.random_flip,
                       self.random_rotation,
                       self.random_gaussian_noise,
                       ]
        return augmentList
