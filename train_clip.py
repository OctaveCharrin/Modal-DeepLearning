import torch
import wandb
import hydra
import os
import clip
from timm.data.auto_augment import rand_augment_transform
from tqdm import tqdm
from torch.utils.data import DataLoader


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    # os.environ['WANDB_API_KEY'] = '045006204280bf2b17bd53dfd35a0ba8e54d00b6'
    # os.environ['WANDB_MODE'] = 'offline'

    logger = wandb.init(project="challenge", name="run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_dataset = datamodule.train_dataset
    class_to_idx = datamodule.dataset.class_to_idx

    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_list[index] = class_name
    
    rand_aug = rand_augment_transform(config_str='rand-m9-n3--mstd0.5')
    train_dataset.transform = rand_aug
    
    train_loader = DataLoader(train_dataset, batch_size=datamodule.batch_size, shuffle=True, num_workers=datamodule.num_workers)
    val_loader = datamodule.val_dataloader()

    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-B/16", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = model.float()
    # print(model)
    # print(len(model.parameters()))


    # for i, params in enumerate(model.parameters()):
    #     print(i)
    #     if i<=250:
    #         params.requires_grad = False

    text = clip.tokenize(class_list).to(device)

    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())

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