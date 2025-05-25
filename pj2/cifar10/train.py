import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import *
from model import *
from log import Logger

from loaders import get_cifar_loader

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=70, metavar='N', help='number of epochs to train (default: 70)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--model', type=str, default='CANet', help='model to use (default: CANet)')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--model_path', type=str, default='cifar10/saved_model/model.pth', help='path to save the model (default: model.pth)')
parser.add_argument('--data_path', type=str, default='../data/', help='path to the dataset (default: ../data/)')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading (default: 4)')
parser.add_argument('--val_rate', type=float, default=0.1, help='validation rate (default: 0.1)')
parser.add_argument('--val_iter', type=int, default=50, help='validation iteration (default: 50)')
parser.add_argument('--report_iter', type=int, default=100, help='report iteration (default: 100)')
parser.add_argument('--device', type=str, default='cuda', help='device to use (default: cuda)')
parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function to use (default: cross_entropy)')
parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer to use (default: adamw)')
parser.add_argument('--lr_scheduler', type=str, default='exponential', help='learning rate scheduler to use (default: exponential)')
parser.add_argument('--weight-decay', type=float, default=1e-3, help='weight decay (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
parser.add_argument('--log_path', type=str, default='runs/canet_exp1', help='path to the log (default: runs/canet_exp1)')

args = parser.parse_args()

def train(model, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    logger = Logger(log_dir=args.log_path)
    model.to(device)
    logger.log_model_graph(model, torch.randn(1, 3, 32, 32).to(device))
    
    batch_size = args.batch_size
    optimizer = optimizer_map[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = lr_schedule_map[args.lr_scheduler](optimizer, **scheduler_params[args.lr_scheduler])
    
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    train_loader, val_loader = get_cifar_loader(root=args.data_path,
                                    batch_size=batch_size,
                                    train=True,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    val_rate=args.val_rate,
                                    custom_transforms=transform
                                    )
    best_acc = 0
    model.train()
    loss_list = {'train':[], 'val':[]}
    acc_list = {'train':[], 'val':[]}
    loss_fn = loss_map[args.loss]
    
    global_step = 0
    
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning Rate :{current_lr}')
        logger.log_lr(current_lr, epoch)
        
        if epoch == int(0.8 * args.epochs):
            optimizer = optimizer_map['sgd'](model.parameters(), lr=current_lr, weight_decay=args.weight_decay, momentum=args.momentum)
            lr_scheduler = lr_schedule_map[args.lr_scheduler](optimizer, **scheduler_params[args.lr_scheduler])
        
        for i, (data, target) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            target = target.to(device)
            data, target_1, target_2, lam = random_cutmix_mixup(data, target, alpha=0.5)
            y_hat = model(data)
            loss = mixup_criterion(loss_fn, y_hat, target_1, target_2, lam)
           # accuracy = (y_hat.argmax(dim=1) == target).float().mean().item()
            accuracy = mixup_accuracy(y_hat, target_1, target_2, lam)
            logger.log_scalars('train', loss.item(), accuracy, global_step)
            
            metrics = {
                   'val_loss': 'not computed',
                   'val_accuracy': 'not computed'
                       }
            
            if i % args.val_iter == 0:
                model.eval()
                val_loss = test_loss(val_loader, loss_map[args.loss], model, device=device)
                val_acc = test_accuracy(val_loader, model, device=device)
                metrics['val_loss'] = val_loss
                metrics['val_accuracy'] = val_acc
                logger.log_scalars('val', val_loss, val_acc, global_step)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    model_dir = os.path.dirname(args.model_path)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    torch.save(model.state_dict(), args.model_path)
                    print(f'Saved model with best val accuracy: {best_acc}')
                model.train()
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % args.report_iter == 0:
                print(f'Epoch {epoch} , Iteration {i}')
                print(f'Recording --> Train Loss: {loss.item()}, Train Accuracy: {accuracy}')
                print(f'Recording --> Validation Loss: {metrics["val_loss"]}, Validation Accuracy: {metrics["val_accuracy"]}')
                print('-----------------------------------')
            
            global_step += 1
            
        logger.log_histograms(model, epoch)
        lr_scheduler.step()
        
        train_loss = test_loss(train_loader, loss_map[args.loss], model, device=device)
        train_acc = test_accuracy(train_loader, model, device=device)
        val_loss = test_loss(val_loader, loss_map[args.loss], model, device=device)
        val_acc = test_accuracy(val_loader, model, device=device)
        
        loss_list['train'].append(train_loss)
        loss_list['val'].append(val_loss)
        acc_list['train'].append(train_acc)
        acc_list['val'].append(val_acc)
        
        logger.log_scalars('epoch_train', train_loss, train_acc, epoch)
        logger.log_scalars('epoch_val', val_loss, val_acc, epoch)
    
    logger.close()
        
    # plot the loss & accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(loss_list['train'], label='Train Loss')
    axes[0].plot(loss_list['val'], label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(acc_list['train'], label='Train Accuracy')
    axes[1].plot(acc_list['val'], label='Validation Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('loss_acc.png')
    plt.show()

if args.model in ['custom_cnn', 'CANet', 'CANet_light', 'CANet_pro', 'CANet_tiny']:
    model = model_map[args.model](input_shape=(3, 32, 32), num_classes=args.num_classes)
else:
    model = model_map[args.model]

if __name__ == '__main__':
    # Set random seed for reproducibility
    set_seed(args.seed)
    train(model, args)