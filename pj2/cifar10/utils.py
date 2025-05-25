import torch
import numpy as np
import random
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_accuracy(data_loader, model, device='cuda'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            
            _, predicted = torch.max(output, 1)

            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    return accuracy

def mixup(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_accuracy(outputs, y_a, y_b, lam):
    preds = outputs.argmax(dim=1)
    acc_a = (preds == y_a).float().mean()
    acc_b = (preds == y_b).float().mean()
    acc = lam * acc_a + (1 - lam) * acc_b
    acc = acc.float().mean().item()
    return acc

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # bbox
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def random_cutmix_mixup(data, target, alpha=0.5, step=[0.3, 0.6]):
    p = random.random()
    if p < step[0]:
        return mixup(data, target, alpha)
    elif p < step[1]:
        return cutmix(data, target, alpha)
    else:
        return data, target, target, 1

def test_loss(data_loader, loss_fn, model, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = loss_fn(output, target)

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

def extract_attention_map(sa_module, feature_tensor):
    # extract attention map from SALayer
    with torch.no_grad():
        return sa_module(feature_tensor)

loss_map = {
    'cross_entropy': torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1),
    'mse': torch.nn.MSELoss(),
    'l1': torch.nn.L1Loss(),
    }

optimizer_map = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'adamw': torch.optim.AdamW,
}

lr_schedule_map = {
    'step': torch.optim.lr_scheduler.StepLR,
    'multi_step': torch.optim.lr_scheduler.MultiStepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

scheduler_params = {
    'step': {'step_size': 6, 'gamma': 0.1},
    'multi_step': {'milestones': [4, 12, 14], 'gamma': 0.1},
    'exponential': {'gamma': 0.96},
    'cosine': {'T_max': 50, 'eta_min': 0},
    'reduce_on_plateau': {'mode': 'min', 'factor': 0.1, 'patience': 10, 'verbose': True},
}