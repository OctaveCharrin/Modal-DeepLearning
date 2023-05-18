import torch
import wandb
import hydra
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from augments.augmentationtransforms import AugmentationTransforms


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    os.environ['WANDB_API_KEY'] = '045006204280bf2b17bd53dfd35a0ba8e54d00b6'
    # os.environ['WANDB_MODE'] = 'offline'

    logger = wandb.init(project="challenge", name=cfg.wandb_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datasetmodule)

    train_dataset = datamodule.train_dataset
    augments = AugmentationTransforms().toList()
    if cfg.no_augmentation :
        augments = augments[:1]
    
    val_loader = datamodule.val_dataloader()




    for epoch in tqdm(range(cfg.num_pseudo_epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for transform in tqdm(augments):
            # Create the dataloader with the right transform
            train_dataset.transform = transform
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
    torch.save(model.state_dict(), cfg.checkpoint_before_pseudo_path)

    # Pseudo Labelling 
    for pseudo_epoch in tqdm(range(cfg.pseudo_epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        # Pseudo labeling
        unlabelled_dataset = datamodule.unlabelled_dataset
        unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=datamodule.batch_size, shuffle=False, num_workers=datamodule.num_workers)
        pseudo_labels = []

        print("Pseudo-labelling...")

        with torch.no_grad():
            for batch in tqdm(unlabelled_loader):
                images, _ = batch
                images = images.to(device)
                preds = model(images)
                probabilities = torch.softmax(preds, dim=1)
                max_probabilities, predicted_labels = torch.max(probabilities, dim=1)

                # Apply a threshold to filter out uncertain predictions
                threshold = 0.9
                confident_predictions = max_probabilities > threshold

                # Assign pseudo-labels to confident predictions
                pseudo_labels.extend(predicted_labels[confident_predictions].tolist())

        # Combine labeled and pseudo-labeled data
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, pseudo_labels])

        # Create a new dataloader with the combined dataset
        combined_loader = DataLoader(combined_dataset, batch_size=datamodule.batch_size, shuffle=True, num_workers=datamodule.num_workers)

        print("Done.")

        for epoch in tqdm(range(cfg.epochs)):
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

    torch.save(model.state_dict(), cfg.checkpoint_path)
    wandb.finish()


if __name__ == "__main__":
    train()