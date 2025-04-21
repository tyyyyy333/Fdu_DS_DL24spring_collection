# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from mynn.op import *

model = nn.models.Model_CNN()
model.load_model(r'codes\best_models_cnn\best_model.pickle')

test_images_path = r'codes\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'codes\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = (test_imgs-test_imgs.min()) / (test_imgs.max() - test_imgs.min())

logits = model(test_imgs)
'''
mats = []
mats.append(model.layers[0].params['W'])
mats.append(model.layers[2].params['W'])

_, axes = plt.subplots(30, 20)
_.set_tight_layout(1)
axes = axes.reshape(-1)
for i in range(600):
        axes[i].matshow(mats[0].T[i].reshape(28,28))
        axes[i].set_xticks([])
        axes[i].set_yticks([])

plt.figure()
plt.matshow(mats[1])
plt.xticks([])
plt.yticks([])
plt.show()
'''

def visualize(model):

    kernel_idx = 0
    for layer in model.layers:
        if isinstance(layer, conv2D):
            W = layer.params['W']
            num_kernels = W.shape[0]
            in_channels = W.shape[1]
            kernel_size = W.shape[2]

            for i in range(num_kernels):
                fig, ax = plt.subplots(1, in_channels, figsize=(15, 5))
                for c in range(in_channels):
                    ax[c].imshow(W[i, c, :, :], cmap='gray')
                    ax[c].axis('off')

                fig.suptitle(f'Conv Layer {kernel_idx}: Kernel {i}')
                plt.show()

            kernel_idx += 1

    layer_idx = 0
    for layer in model.layers:
        if isinstance(layer, Linear):
            W = layer.params['W']
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(W, cmap='viridis', aspect='auto')
            ax.set_title(f'Linear Layer {layer_idx} Weights')
            ax.set_xlabel('Output units')
            ax.set_ylabel('Input units')
            plt.colorbar(ax.imshow(W, cmap='viridis', aspect='auto'))
            plt.show()

            layer_idx += 1

visualize(model)