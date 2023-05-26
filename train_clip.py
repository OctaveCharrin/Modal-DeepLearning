import torch
import wandb
import hydra
import os
import clip
import torch.optim as optim
from timm.data.auto_augment import rand_augment_transform
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    # os.environ['WANDB_API_KEY'] = '045006204280bf2b17bd53dfd35a0ba8e54d00b6'
    os.environ['WANDB_MODE'] = 'offline'

    learning_rate = 1e-7

    logger = wandb.init(project="challenge", name=cfg.wandb_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    simple_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.ToTensor(),
            ])
    rand_aug = rand_augment_transform(config_str='rand-m9-n3--mstd0.5',hparams={'img_mean': (0.485, 0.456, 0.406), 'img_std':(0.229, 0.224, 0.225)})

    traindir = os.getcwd() + '/datasetV2/train_val'
    valdir = os.getcwd() + '/datasetV2/val_train'

    train_dataset = ImageFolder(traindir, transform=rand_aug)
    val_dataset = ImageFolder(valdir, transform=simple_transform)
    class_to_idx = train_dataset.class_to_idx

    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_list[index] = class_name
    
    
    train_loader = DataLoader(train_dataset, batch_size=datamodule.batch_size, shuffle=True, num_workers=datamodule.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=datamodule.batch_size, shuffle=False, num_workers=datamodule.num_workers)

    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-B/16", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.float()

    for name, param in model.named_parameters():
        if name.startswith('transfomer'):
            param.requires_grad = False

    # interval = range(250, 1000)
    # for i, params in enumerate(model.parameters()):
    #     if i in interval:
    #         params.requires_grad = False

    text = clip.tokenize(class_list).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

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
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        path = os.getcwd() + f"/checkpoints/freezeclip_training_checkpoint{epoch}.pt"
        torch.save(model.state_dict(), path)

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
    torch.save(model.state_dict(), cfg.checkpoint_path)
    wandb.finish()


if __name__ == "__main__":
    train()