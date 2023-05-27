import torch
import hydra
import os
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

@hydra.main(config_path="configs", config_name="config", version_base=None)
def test(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the desired model

    model, preprocess = clip.load("ViT-B/16", device=device)

    checkpoints_path =  os.path.join(cfg.root_dir, 'checkpoints')
    path = os.path.join(checkpoints_path, 'RESUME_clip16_simple_finetune_mlpunfrozen_checkpoint_epoch_18.pt')
    checkpoint = torch.load(path)
    print("loading your model from ", path, "...")
    model.load_state_dict(checkpoint)
    print("loading completed.")
    
    model.eval()

    simple_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224),antialias=None),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    print("creating dataloader...")
    valdir = os.path.join(cfg.data_dir, 'val_train')
    val_dataset = ImageFolder(valdir, transform=simple_transform)
    class_to_idx = val_dataset.class_to_idx

    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_list[index] = class_name

    val_loader = DataLoader(val_dataset, batch_size=datamodule.batch_size, shuffle=False, num_workers=datamodule.num_workers)
    print("done.")

    text = clip.tokenize(class_list).to(device)

    # Compute the accuracy of the model
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            predictions, _ = model(images, text)
            _, predicted_labels = torch.max(predictions, 1)
            num_correct += (predicted_labels == labels).sum().item()
            num_samples += labels.size(0)
    accuracy = num_correct / num_samples
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    test()