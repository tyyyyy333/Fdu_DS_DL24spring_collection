import numpy as np
import torch
from utils import *
from model import *
from loaders import get_cifar_loader

import argparse

def test(model, test_data_loader, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"test accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Test a model on CIFAR-10')
    parser.add_argument('--model', type=str, default='CANet', help='model architecture')
    parser.add_argument('--model_path', type=str, default='saved_model/CANet.pth', help='path to the model')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for testing (default: cuda)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_data_loader = get_cifar_loader(root='../data/',
                                        batch_size=args.batch_size,
                                        train=False,
                                        shuffle=False,
                                        custom_transforms=None,
                                        num_workers=args.num_workers
                                        )

    if args.model in ['custom_cnn', 'CANet', 'CANet_light', 'CANet_pro', 'CANet_tiny']:
        model = model_map[args.model](input_shape=(3, 32, 32), num_classes=args.num_classes)
    else:
        model = model_map[args.model]

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    else:
        raise ValueError("Model path is required to load the model.")

    test(model, test_data_loader)