import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

num_workers = 4
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_and_log(models, model_names, optimizers, criterion, train_loader, val_loader, scheduler=None, epochs_n=100):
    writers = {name: SummaryWriter(log_dir=os.path.join("runs", "Comparison")) for name in model_names}
    all_losses, all_grads, all_val_accs, all_train_accs = {}, {}, {}, {}

    for model, name, optimizer in zip(models, model_names, optimizers):
        model.to(device)
        losses_list, grads, val_accs, train_accs = [], [], [], []

        for epoch in tqdm(range(epochs_n), unit='epoch'):
            model.train()
            epoch_loss = 0
            epoch_grads = []

            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                prediction = model(x)
                loss = criterion(prediction, y)
                loss.backward()
                optimizer.step()

                grad_values = model.classifier[-1].weight.grad.detach().cpu().numpy().flatten()
                grad_max = grad_values.max()
                grad_min = grad_values.min()
                epoch_grads.append((grad_max, grad_min))

                epoch_loss += loss.item()
                losses_list.append(loss.item())
                writers[name].add_scalar(f'Loss/train_step_of_{name}', loss.item(), epoch * len(train_loader) + step)

            avg_epoch_loss = epoch_loss / len(train_loader)
            grads.append(epoch_grads)

            train_acc = get_accuracy(model, train_loader)
            val_acc = get_accuracy(model, val_loader)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            writers[name].add_scalar(f'Loss/train_epoch_of_{name}', avg_epoch_loss, epoch)
            writers[name].add_scalar(f'Accuracy/epoch_train_of_{name}', train_acc, epoch)
            writers[name].add_scalar(f'Accuracy/epoch_val_of_{name}', val_acc, epoch)

        all_losses[name] = losses_list
        all_grads[name] = grads
        all_val_accs[name] = val_accs
        all_train_accs[name] = train_accs
        writers[name].close()

    return all_losses, all_grads, all_train_accs, all_val_accs


def log_grad_range_to_tensorboard(grads_dict, epochs, model_names):
    writer = SummaryWriter(log_dir=os.path.join("runs", "Comparison"))
    max_avg_dict = {name : [] for name in model_names}
    min_avg_dict = {name : [] for name in model_names}
    for name in model_names:

        for epoch in range(epochs):
            step_grads = grads_dict[name][epoch]
            max_vals = [g[0] for g in step_grads]
            min_vals = [g[1] for g in step_grads]

            max_avg_dict[name].append(np.mean(max_vals))
            min_avg_dict[name].append(np.mean(min_vals))

            for step, (gmax, gmin) in enumerate(step_grads):
                global_step = epoch * len(step_grads) + step
                writer.add_scalar(f'{name}/GradMax_Step', gmax, global_step)
                writer.add_scalar(f'{name}/GradMin_Step', gmin, global_step)
                writer.add_scalar(f'{name}/GradMaxMinusMin_Step', gmax - gmin, global_step)
                
        for epoch in range(epochs):
            writer.add_scalar(f'{name}/GradMax_EpochAvg', max_avg_dict[name][epoch], epoch)
            writer.add_scalar(f'{name}/GradMin_EpochAvg', min_avg_dict[name][epoch], epoch)
            writer.add_scalar(f'{name}/GradRange_EpochAvg', max_avg_dict[name][epoch] - min_avg_dict[name][epoch], epoch)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(epochs))
    colors = ['#F00000', '#00F0F0']
    for i, name in enumerate(model_names):
        c = colors[i % len(colors)]
        ax.plot(x, max_avg_dict[name], label=f"Grad Max of {name}", color=c, alpha=0.6)
        ax.plot(x, min_avg_dict[name], label=f"Grad Min {name}", color=c, alpha=0.6)
        ax.fill_between(x, min_avg_dict[name], max_avg_dict[name], color=c, alpha=0.3, label=f"Grad Range of {name}")
    
    ax.set_title(f"Gradient Range Over Epochs: {name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Value")
    ax.legend()
    fig.tight_layout()
    writer.add_figure(f'GradRange_Figure_comparison', fig, global_step=epochs - 1)
    plt.close(fig)

    writer.close()

def log_loss_range_to_tensorboard(loss_dict):
    writer = SummaryWriter(log_dir=os.path.join("runs", "Comparison"))

    bn = []
    nobn = []
    for key, values in loss_dict.items():
        if 'BatchNorm' in key:
            bn.append(values)
        else:
            nobn.append(values)

    bn = np.array(bn)[:, :]
    nobn = np.array(nobn)[:, :]

    steps = bn.shape[1]
    x = np.arange(steps)

    bn_max = np.max(bn, axis=0)
    bn_min = np.min(bn, axis=0)
    nobn_max = np.max(nobn, axis=0)
    nobn_min = np.min(nobn, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    c1 = '#F00000'
    c2 = '#00F0F0'
    ax.plot(x, bn_max, label="Max Loss of VGG_A_BatchNorm", color=c1, alpha=0.9)
    ax.plot(x, bn_min, label="Min Loss of VGG_A_BatchNorm", color=c1, alpha=0.9)
    ax.fill_between(x, bn_min, bn_max, color=c1, alpha=0.3, label="Range of VGG_A_BatchNorm")

    ax.plot(x, nobn_max, label="Max Loss of VGG_A", color=c2, alpha=0.6)
    ax.plot(x, nobn_min, label="Min Loss of VGG_A", color=c2, alpha=0.6)
    ax.fill_between(x, nobn_min, nobn_max, color=c2, alpha=0.3, label="Range of VGG_A")
    ax.set_xlim(xmin = -200, xmax = steps+1000)
    ax.set_title("Step-wise Loss Range Comparison (Different Learning Rates)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss Value")
    ax.legend()
    fig.tight_layout()

    writer.add_figure('LossRange_Figure_comparison', fig, global_step=steps - 1)
    plt.close(fig)
    writer.close()



if __name__ == '__main__':
    epoch = 50
    set_random_seeds(seed_value=2020, device=device)
    lr_list = [0.001, 0.002, 0.0005, 0.0001]
    losses = {}
    for lr in lr_list:
        model = VGG_A()
        model_bn = VGG_A_BatchNorm()
        optimizer_plain = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)

        models = [model, model_bn]
        model_names = ["VGG_A"+str(lr), "VGG_A_BatchNorm"+str(lr)]
        optimizers = [optimizer_plain, optimizer_bn]
        criterion = nn.CrossEntropyLoss()
        losses_dict, grads_dict, train_accs_dict, val_accs_dict = train_and_log(
            models, model_names, optimizers, criterion, train_loader, val_loader, epochs_n=epoch)
        
        losses.update(losses_dict)
    
    log_grad_range_to_tensorboard(grads_dict, epochs=epoch, model_names=model_names)
    log_loss_range_to_tensorboard(losses)
    print("Done.")
