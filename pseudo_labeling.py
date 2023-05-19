import torch
import wandb
import hydra
import os
import numpy as np

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset, Subset

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    # os.environ['WANDB_API_KEY'] = '045006204280bf2b17bd53dfd35a0ba8e54d00b6'
    os.environ['WANDB_MODE'] = 'offline'

    logger = wandb.init(project="challenge", name="pseudo_labeling")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model specs
    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    # Create datasets and dataloader
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    train_dataset = datamodule.train_dataset
    val_loader = datamodule.val_dataloader()

    # First training method
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
            epoch_num_correct += ((preds.argmax(1) == labels).sum().detach().cpu().numpy())
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )

        epoch_acc, epoch_loss = evaluate(model, loss_fn=loss_fn, dataloader=val_loader, device=device)

        logger.log(
            {
                "epoch": epoch,
                "val_loss_epoch": epoch_loss,
                "val_acc": epoch_acc,
            }
        )
    # End of first training

    # Begin pseudo-labeling
    unlabeled_dataset = UnlabelledDataset(cfg.datasetmodule.unlabeled_dataset_path, transform=hydra.utils.instantiate(cfg.datamodule.train_transform))
    
    # For test 
    num_images = 100
    indices = torch.randperm(len(unlabeled_dataset))
    selected_indices = indices[:num_images]
    unlabeled_dataset = Subset(unlabeled_dataset, selected_indices)

    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    for pseudo_epoch in range(cfg.pseudo_epochs):
        threshold = 0.7 + (0.95-0.7)*np.exp(2*(pseudo_epoch/(cfg.pseudo_epochs+1)-1))
        pseudo_dataset = generate_pseudo_dataset(model, unlabeled_dataloader, train_dataset, threshold, device)
        pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)

        train_and_eval(model, pseudo_dataloader, val_loader, loss_fn, optimizer, cfg.epochs, device, logger, "pseudo_label")

    torch.save(model.state_dict(), cfg.checkpoint_path)
    wandb.finish()


class UnlabelledDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.folder_path, self.file_list[index])
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, -1


def generate_pseudo_dataset(model, dataloader, combined, threshold, device):
    print('Generating pseudo labels ...')
    model.eval()
    pseudo_dataset = combined
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            images, _ = batch
            images = images.to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            max_probas, predictions = torch.max(outputs, dim=1)

            mask = max_probas > threshold
            tokens, labels = images[mask], predictions[mask]
            tokens = tokens.to('cpu')
            labels = labels.to('cpu')
            dataset = TensorDataset(tokens, labels)
            pseudo_dataset = ConcatDataset([pseudo_dataset, dataset])
    model.train()
    print('Done.')
    return pseudo_dataset


def train_and_eval(model, train_loader, val_loader, loss_fn, optimizer, epochs, device, logger, data_name):
    for epoch in tqdm(range(epochs)):
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
            epoch_num_correct += ((preds.argmax(1) == labels).sum().detach().cpu().numpy())
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                f"{data_name}_train_loss_epoch": epoch_loss,
                f"{data_name}_train_acc": epoch_acc,
            }
        )

        epoch_acc, epoch_loss = evaluate(model, loss_fn=loss_fn, dataloader=val_loader, device=device)

        logger.log(
            {
                f"{data_name}_epoch": epoch,
                f"{data_name}_val_loss_epoch": epoch_loss,
                f"{data_name}_val_acc": epoch_acc,
            }
        )


def evaluate(model, loss_fn, dataloader, device):
    '''
    Calculate the accuracy and loss
    '''
    epoch_loss = 0
    epoch_num_correct = 0
    num_samples = 0
    for _, batch in enumerate(dataloader):
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

    return epoch_acc, epoch_loss


def collate_fn(batch):
    images, labels = zip(*batch)
    if isinstance(labels[0], int):
        labels = torch.LongTensor(labels)
    else:
        labels = torch.stack([torch.LongTensor([label.item()]) for label in labels])
    return torch.stack(images), labels


if __name__ == "__main__":
    train()