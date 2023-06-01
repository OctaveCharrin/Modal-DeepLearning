import torch
import wandb
import hydra
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset


@hydra.main(config_path="configs", config_name="pseudo_labeling_config", version_base=None)
def train(cfg):

    os.environ['WANDB_MODE'] = 'offline'

    logger = wandb.init(project="challenge", name="run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(cfg.model, device=device).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datasetmodule)

    train_dataset = datamodule.train_dataset
    train_loader = datamodule.train_dataloader()    
    val_loader = datamodule.val_dataloader()

    # Training
    for epoch in tqdm(range(cfg.training_epochs)):
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

    # Pseudo Labelling 
    unlabeled_dataset = datamodule.unlabeled_dataset

    for pseudo_epoch in tqdm(range(cfg.pseudo_labeling_epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        # In case you need an evolving threshold
        x = pseudo_epoch/(cfg.pseudo_labeling_epochs)
        threshold = 0.7 + (0.9 - 0.7) * np.exp(2*(x-1))

        pseudo_labels = generate_pseudo_labels(model, unlabeled_dataset, device, threshold=0.1)

        if pseudo_labels:
            print(f"Pseudo-labeled samples in epoch {pseudo_epoch+1}: {len(pseudo_labels)}")
            # Separate the inputs and labels into two tensors
            pseudo_inputs, pseudo_labels = zip(*pseudo_labels)
            pseudo_inputs = torch.stack(pseudo_inputs)
            pseudo_labels = torch.tensor(pseudo_labels)
            pseudo_labeled_dataset = TensorDataset(pseudo_inputs, pseudo_labels)
        else:
            continue
        combined_dataset = ConcatDataset([train_dataset, pseudo_labeled_dataset])
        print(f'The dataset now contains {len(combined_dataset)} images.')
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

        for epoch in tqdm(range(cfg.epochs_per_pass)):
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
                "pseudo_epoch": pseudo_epoch,
                "pseudo_val_loss_epoch": epoch_loss,
                "pseudo_val_acc": epoch_acc,
            }
        )
        torch.save(model.state_dict(), os.path.join(cfg.checkpoint_path, f'{cfg.checkpoint_name}_pseudoepoch_{pseudo_epoch}.pt'))

    torch.save(model.state_dict(), os.path.join(cfg.checkpoint_path, f'{cfg.checkpoint_name}_end.pt'))
    wandb.finish()


def generate_pseudo_labels(model, dataset, device, threshold):

    pseudo_labels = []
    print(f"Generating pseudo-labeled samples from {len(dataset)} images...")

    data_loader = DataLoader(dataset, batch_size = 100, shuffle = True, num_workers = 0)

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits_per_image = model(inputs)

            probabilities = torch.softmax(logits_per_image, dim=1)
            max_probabilities, max_indices = torch.max(probabilities, dim=1)

            for input, max_probability, max_index in zip(inputs, max_probabilities, max_indices):
                if max_probability.item() > threshold:
                    # Move the tensors back to the CPU before appending to the list
                    input = input.to("cpu")
                    max_index = max_index.to("cpu")
                    pseudo_labels.append((input, torch.tensor([max_index.item()], dtype=torch.long)))

    model.train()
    return pseudo_labels


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels


if __name__ == "__main__":
    train()