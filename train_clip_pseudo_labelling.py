import torch
import wandb
import hydra
import os
import clip
import timm
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
from augments.augmentationtransforms import AugmentationTransforms


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

@hydra.main(config_path="configs", config_name="configV2", version_base=None)
def train(cfg):

    # os.environ['WANDB_API_KEY'] = '045006204280bf2b17bd53dfd35a0ba8e54d00b6'
    # os.environ['WANDB_MODE'] = 'offline'

    logger = wandb.init(project="challenge", name=cfg.wandb_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datasetmodule)

    train_dataset = datamodule.train_dataset
    class_to_idx = datamodule.dataset.class_to_idx

    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_list[index] = class_name
    
    val_loader = datamodule.val_dataloader()


    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        train_loader = DataLoader(train_dataset, batch_size=datamodule.batch_size, shuffle=True, num_workers=datamodule.num_workers)

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
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
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
    # torch.save(model.state_dict(), cfg.checkpoint_before_pseudo_path)

    # Pseudo Labelling 
    combined_dataset = train_dataset
    unlabeled_dataset = datamodule.unlabeled_dataset
    indexes = list(range(len(unlabeled_dataset)))
    random.shuffle(indexes)

    for pseudo_epoch in tqdm(range(cfg.pseudo_epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        # Pseudo labeling

        # In case you need a threshold
        # x = pseudo_epoch/(cfg.pseudo_epochs)
        # threshold = 0.7 + (0.9 - 0.7) * np.exp(2*(x-1))

        pseudo_labels, indexes = generate_clip_pseudo_labels(class_list, indexes, unlabeled_dataset, device, threshold=0.95, time = pseudo_epoch+1)

        if pseudo_labels:
            print(f"Pseudo-labeled samples in epoch {pseudo_epoch+1}: {len(pseudo_labels)}")

            # Separate the inputs and labels into two tensors
            pseudo_inputs, pseudo_labels = zip(*pseudo_labels)
            pseudo_inputs = torch.stack(pseudo_inputs)
            pseudo_labels = torch.tensor(pseudo_labels)
            pseudo_labeled_dataset = TensorDataset(pseudo_inputs, pseudo_labels)

            # def collate_fn(batch):
            #     inputs, labels = zip(*batch)
            #     inputs = torch.stack(inputs)
            #     labels = torch.tensor(labels, dtype=torch.long)
            #     return inputs, labels

        combined_dataset = ConcatDataset([combined_dataset, pseudo_labeled_dataset])
        print(f'The dataset now contains {len(combined_dataset)} images.')
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

        for epoch in tqdm(range(cfg.list_pseudo_epochs[pseudo_epoch])):
            for _, batch in enumerate(combined_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = loss_fn(preds, labels)
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
                    "pseudo_train_loss_epoch": epoch_loss,
                    "pseudo_train_acc": epoch_acc,
                }
            )
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0

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
                "pseudo_epoch": pseudo_epoch*epoch,
                "pseudo_val_loss_epoch": epoch_loss,
                "pseudo_val_acc": epoch_acc,
            }
        )
        checkpointn = 3
        if pseudo_epoch%checkpointn==0:
            path = os.path.join(cfg.root_dir, f'checkpoints/24-05-2023-vit_clip_list_checkpoit{pseudo_epoch//checkpointn}.pt')
            print('saving model at:', path)
            torch.save(model.state_dict(), path)

    torch.save(model.state_dict(), cfg.checkpoint_path)
    wandb.finish()


def generate_clip_pseudo_labels(class_list, indexes, dataset, device, threshold, time):
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(class_list).to(device)

    pseudo_labels = []

    time = min(5, time)

    print(f"Generating pseudo labels from {time*1000} images...")

    index = indexes[:1000*time]
    indexes = indexes[1000*time:]

    data_loader = DataLoader(Subset(dataset, index), batch_size = 100, shuffle = True, num_workers = 0)

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits_per_image, logits_per_text = model(inputs, text)

            probabilities = torch.softmax(logits_per_image, dim=1)
            max_probabilities, max_indices = torch.max(probabilities, dim=1)

            for input, max_probability, max_index in zip(inputs, max_probabilities, max_indices):
                if max_probability.item() > threshold:
                    # Move the tensors back to the CPU before appending to the list
                    input = input.to("cpu")
                    max_index = max_index.to("cpu")
                    pseudo_labels.append((input, torch.tensor([max_index.item()], dtype=torch.long)))

    model.train()
    return pseudo_labels, indexes

if __name__ == "__main__":
    train()