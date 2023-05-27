import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import hydra
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import clip
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    # Assuming you have already trained your model and have a data loader 'loader'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = hydra.utils.instantiate(cfg.datamodule)


    valdir = os.path.join(cfg.data_dir, 'val_train')
    simple_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224),antialias=None),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    val_dataset = ImageFolder(valdir, transform=simple_transform)

    val_loader = DataLoader(val_dataset, batch_size=datamodule.batch_size, shuffle=True, num_workers=datamodule.num_workers)

    model, preprocess = clip.load("ViT-B/16", device=device)

    fname = 'RESUME_clip16_wd10_simple_allmlpunfrozen_checkpoint_final'

    checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    path = os.path.join(checkpoints_path, f'{fname}.pt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    # Set the model to evaluation mode
    model.eval()

    class_to_idx = val_dataset.class_to_idx

    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_list[index] = class_name.lower()
    text = clip.tokenize(class_list).to(device)

    # Initialize empty lists for predictions and labels
    all_predictions = []
    all_labels = []

    # Iterate over the data loader
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            predictions, _ = model(images, text)
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

    pathjpg = os.path.join(cfg.root_dir, f'confmat_{fname}.jpg')
    pathtxt = os.path.join(cfg.root_dir, f'confmat_{fname}.txt')

    # Print the confusion matrix
    save_confmat(confusion_mat, class_list, pathjpg)
    save_txt_file(pathtxt, confusion_mat, class_list, pathjpg)


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

def save_txt_file(path, matrix, class_names,fname):
    my_dict = {'matrix':matrix, 'class_dict':class_names, 'fname':fname}
    with open(path, 'w') as file:
        json.dump(my_dict, file)


if __name__ == '__main__':
    main()