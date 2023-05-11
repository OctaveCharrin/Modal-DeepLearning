import torch
import hydra
from tqdm import tqdm

@hydra.main(config_path="configs", config_name="config", version_base=None)
def test(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the desired model
    print("loading your model from ", cfg.checkpoint_path, "...")
    model = hydra.utils.instantiate(cfg.model, frozen=False).to(device)
    state_dict = torch.load(cfg.checkpoint_path)
    model.load_state_dict(state_dict)
    print("loading completed.")
    
    model.eval()

    print("creating dataloader...")
    datasetmodule = hydra.utils.instantiate(cfg.datasetmodule)
    test_loader = datasetmodule.train_dataloader()
    print("done.")

    # Compute the accuracy of the model
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels = batch
            predictions = model(images)
            _, predicted_labels = torch.max(predictions, 1)
            num_correct += (predicted_labels == labels).sum().item()
            num_samples += labels.size(0)
    accuracy = num_correct / num_samples
    print(f"Accuracy: {accuracy}")

    model.train()


if __name__ == "__main__":
    test()