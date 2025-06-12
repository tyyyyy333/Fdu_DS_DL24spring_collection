import argparse
import torch
import os

from loaders import get_cifar_loader
from log import Logger
from model import *
from test import test

from torchinfo import summary
from io import StringIO
import sys

def log_model_summary_to_tensorboard(writer, model, input_size, global_step=0, tag='ModelSummary', device='cpu'):
    model_summary_str = get_model_summary(model, input_size, device=device)
    writer.log_text(tag, f'```\n{model_summary_str}\n```', global_step)

def get_model_summary(model, input_size, device='cpu'):
    # redirect stdout to capture summary output
    buffer = StringIO()
    sys.stdout = buffer
    summary(model, input_size=input_size, device=device)
    sys.stdout = sys.__stdout__
    return buffer.getvalue()

def per_class_accuracy(model, dataloader, logger, class_names=None, device='cuda'):
    model.eval()
    num_classes = 10 if class_names is None else len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            for i in range(len(y)):
                label = y[i].item()
                class_total[label] += 1
                if preds[i] == label:
                    class_correct[label] += 1

    for i in range(num_classes):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        cname = class_names[i] if class_names else str(i)
        print(f"类别 {cname} 准确率: {acc:.2f}%")
        logger.log_text('PerClassAccuracy', f"类别 {cname} 准确率: {acc:.2f}%", step=0)
        
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 模型加载
    if args.model in ['custom_cnn', 'CANet', 'CANet_light', 'CANet_pro', 'CANet_tiny']:
        model = model_map[args.model](input_shape=(3, 32, 32), num_classes=10)
    else:
        model = model_map[args.model]

    
    # 权重加载
    if args.model_path and os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"模型权重加载完成：{args.model_path}")
    model.to(device)
    logger = Logger(log_dir=args.log_path)

    log_model_summary_to_tensorboard(logger, model, (1, 3, 32, 32), global_step=0, tag='args.model', device=args.device)

    # 数据加载（只用测试集）
    test_loader = get_cifar_loader(root=args.data_root, train=False,
                                   batch_size=args.batch_size,
                                   custom_transforms=None,
                                   shuffle=False,
                                   num_workers=args.num_workers)
    per_class_accuracy(model, test_loader, logger)
    # 运行测试
    acc = test(model, test_loader)
    logger.log_scalars(phase='Test', acc=acc, step=0, loss=0)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Model and Test")
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--model_path', type=str, default=None, help='path to the model weights')
    parser.add_argument('--log_path', type=str, default='runs/vis_model', help='TensorBoard log directory')
    parser.add_argument('--data_root', type=str, default='../data/', help='path to the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for testing (default: cuda)')

    args = parser.parse_args()
    main(args)
