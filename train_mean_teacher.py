import torch
import wandb
import hydra
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from augments.augmentationtransforms import AugmentationTransforms
from data.twostreambatchsampler import TwoStreamBatchSampler
from models.mean_teacher import MeanTeacherModel
import torch.nn as nn
import losses.losses as losses
import numpy as np

import data.datasampler as data
import torchvision.transforms as transforms
import torchvision

global_step = 0

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):

    global global_step

    # os.environ['WANDB_API_KEY'] = '045006204280bf2b17bd53dfd35a0ba8e54d00b6'
    # os.environ['WANDB_MODE'] = 'offline'

    logger = wandb.init(project="challenge", name=cfg.wandb_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = hydra.utils.instantiate(cfg.datasetmodule)
    train_transform = data.TransformTwice(transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.Resize((224,224)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]))

    dataset = torchvision.datasets.ImageFolder(cfg.datamodule.train_dataset_path, train_transform)

    labelled_idxs = range(720)
    unlabelled_idxs = range(720, len(dataset))

    batch_sampler = data.TwoStreamBatchSampler(unlabelled_idxs, labelled_idxs, cfg.batch_size, cfg.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True)
    
    val_loader = datamodule.val_dataloader()

    # Intializing the model
    model = MeanTeacherModel(cfg.model.num_classes, frozen=False, no_grad = False).to(device)
    ema_model = MeanTeacherModel(cfg.model.num_classes, cfg.model.frozen, no_grad = True).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), 1e-5,
                                momentum=0.9,
                                weight_decay=1e-4)

    for epoch in tqdm(range(cfg.epochs)):
        running_loss = train(cfg, train_loader, model, ema_model, optimizer, epoch, device)

        prec1 = validate(val_loader, model, device)
        ema_prec1 = validate(val_loader, ema_model, device)

        print("loss: ", running_loss)
        print('Accuracy of the Student network on the test images: %d %%' % (
            prec1))
        print('Accuracy of the Teacher network on the test images: %d %%' % (
            ema_prec1))
        
        logger.log({
            "epoch":epoch,
            "loss":running_loss,
            "student acc": prec1,
            "teacher acc": ema_prec1
        })


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(cfg, train_loader, model, ema_model, optimizer, epoch, device):
    global global_step

    running_loss = 0.0

    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=data.NO_LABEL).to(device)

    consistency_criterion = losses.softmax_mse_loss

    model.train()
    ema_model.train()

    for i, ((input, ema_input), target) in enumerate(train_loader):

        if (input.size(0) != cfg.batch_size):
            continue

        input_var = input.to(device)
        target_var = target.to(device)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(data.NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        model_out = model(input_var)

        class_loss = class_criterion(model_out, target_var) / minibatch_size
        
        with torch.no_grad():
            ema_input_var = torch.autograd.Variable(ema_input)
            ema_input_var = ema_input_var.to(device)

        ema_model_out = ema_model(ema_input_var)

        ema_logit = ema_model_out

        ema_logit = torch.autograd.Variable(ema_logit.detach().data, requires_grad=False)

        # Consistency loss and loss
        consistency_weight = get_current_consistency_weight(epoch + 1)
        consistency_loss = consistency_weight * consistency_criterion(model_out, ema_logit) / cfg.batch_size
        
        loss = class_loss + consistency_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.data[0])


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        update_ema_variables(model, ema_model, 0.99, global_step)

        # print statistics
        running_loss += loss.item()

        if i % 20 == 19:    # print every 20 mini-batches
            print('[Epoch: %d, Iteration: %5d] loss: %.5f' %
                (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

    return running_loss


def validate(eval_loader, model, device):

    model.eval()
    total =0
    correct = 0
    for i, (input, target) in enumerate(eval_loader):

        with torch.no_grad():
            input_var = input.to(device)
            target_var = target.to(device)

            output1 = model(input_var)

            _, predicted = torch.max(output1, 1)
            total += target_var.size(0)
            correct += (predicted == target_var).sum().item()

    return 100 * correct / total

def get_current_consistency_weight(epoch):

    rampup_length = 5
    current = np.clip(epoch, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    sigmoid_rampup = float(np.exp(-5.0 * phase * phase))

    return 12.5 * sigmoid_rampup

if __name__ =="__main__":
    main()