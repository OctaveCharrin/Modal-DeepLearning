import torch
import hydra
from tqdm import tqdm
import clip

@hydra.main(config_path="configs", config_name="config", version_base=None)
def test(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_dataset = datamodule.train_dataset
    class_to_idx = datamodule.dataset.class_to_idx

    class_list = list(range(48))
    for  (class_name, index) in class_to_idx.items():
        class_list[index] = class_name
    
    text = clip.tokenize(class_list).to(device)

    # load the desired model

    
    model, preprocess = clip.load("ViT-B/32", device=device)
    if cfg.resume:
        print("loading your model from ", cfg.checkpoint_path, "...")
        state_dict = torch.load(cfg.checkpoint_path)
        model.load_state_dict(state_dict)
        print("loading completed.")
    
    model.float()
    model.eval()

    datasetmodule = hydra.utils.instantiate(cfg.datamodule)
    test_loader = datasetmodule.val_dataloader()

    # Compute the accuracy of the model
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits_per_image, logits_per_text = model(images, text)
            num_correct += (
                (logits_per_image.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
    accuracy = num_correct / num_samples
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    test()