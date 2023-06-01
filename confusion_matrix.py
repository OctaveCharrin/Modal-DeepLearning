import torch
import numpy as np
import hydra
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    val_loader = datamodule.val_dataloader()

    class_list = sorted(os.listdir(cfg.dataset.train_path))


    model = hydra.utils.instantiate(cfg.model, device=device).to(device)
    
    # Load the desired model
    fname = 'REPORT_clip16_lr5e-8_wd.01_simple_allunfroz_namechgV2_chckpt_15'
    
    path = os.path.join(cfg.checkpoint_path, f'{fname}.pt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)

    model.eval()

    # Initialize empty lists for predictions and labels
    all_predictions = []
    all_labels = []

    # Iterate over the data loader
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            _, predicted_labels = torch.max(predictions, 1)
            # Append predictions and labels to the lists
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert predictions and labels to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(labels, predictions)
    print(confusion_mat)

    # Save the confusion matrix figure
    pathjpg = os.path.join(cfg.root_dir, f'CONFMAT_{fname}.jpg')
    save_confmat(confusion_mat, class_list, pathjpg)


def save_confmat(matrix, class_dict, fname):
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as per your preference

    heatmap = ax.pcolor(matrix[::-1], cmap='Blues')  # Reverse the matrix using [::-1]

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the ticks at the edges of each cell
    ax.set_xticks(np.arange(matrix.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(matrix.shape[0])+0.5, minor=False)

    # Add labels to the ticks and tilt them by 45 degrees
    ax.set_xticklabels([class_dict[i] for i in range(matrix.shape[1])], ha='center', rotation=90)  # Use class_dict for x-axis labels
    ax.set_yticklabels([class_dict[i] for i in range(matrix.shape[0])][::-1], va='center')  # Use class_dict for y-axis labels

    # Move the x-axis labels to the top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.tight_layout()  # Adjust subplot parameters to fit the figure area

    plt.savefig(fname)


if __name__ == '__main__':
    main()