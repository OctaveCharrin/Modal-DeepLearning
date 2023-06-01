import torch
import hydra
import os
import numpy as np
from tqdm import tqdm

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    val_dataloader = datamodule.val_dataloader()

    # File were the model is stored
    fname = ''

    path = os.path.join(cfg.checkpoint_path, f'{fname}.pt')
    # checkpoint = torch.load(path)

    best = (0,0) # Best accuracy and the corresponding alpha

    (start, end) = (0, 1)
    point_number = 11

    alphas, accs = [], []

    for alpha in np.linspace(start, end, point_number, endpoint=True):

        zeroshot = hydra.utils.instantiate(cfg.model, device=device).to(device)
        finetuned = hydra.utils.instantiate(cfg.model, device=device).to(device)

        # finetuned.load_state_dict(checkpoint)

        theta_0 = zeroshot.state_dict()
        theta_1 = finetuned.state_dict()

        theta = {key : (1-alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()}

        finetuned.load_state_dict(theta)
        acc = eval(finetuned, val_dataloader, device)

        alphas.append(alpha)
        accs.append(accs)

        print(f'accuracy for alpha = {alpha}: ', acc)
        if acc > best[0]:
            best = (acc, alpha)
    print(best)


def eval(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            logits_per_image = model(images)
            num_correct += (
                (logits_per_image.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        acc = num_correct / num_samples
    return acc


if __name__ == '__main__':
    main()