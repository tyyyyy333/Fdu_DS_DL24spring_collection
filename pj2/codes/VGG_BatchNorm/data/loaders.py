"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchvision.datasets as datasets



class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root='../data/',
                     batch_size=128,
                     train=True,
                     shuffle=True,
                     custom_transforms = None,
                     num_workers=4,
                     n_items=-1,
                     val_rate=0.1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    if custom_transforms is None:
        data_transforms = transforms.Compose(
            [transforms.ToTensor(),
            normalize])
    else:
        data_transforms = custom_transforms

    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)
        
    if train:
        train_size = int(len(dataset) * (1 - val_rate))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
        return train_loader, val_loader
    
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == '__main__':
    train_loader, val_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break