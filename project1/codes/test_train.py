# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = 'codes\dataset\MNIST\\train-images-idx3-ubyte.gz'
train_labels_path = 'codes\dataset\MNIST\\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
        

train_imgs = train_imgs[idx].reshape(-1, 1, 28, 28)   # reshape if model is CNN
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

#transformer = nn.transforms.ImageTransformer(train_imgs)

#transforms = [
#    lambda: transformer.translate(3, 3),
#    lambda: transformer.rotate(15),
#    lambda: transformer.random_resize((0.95, 1.05))
#]

#augmented_images = transformer.pipeline(transforms, aug_parallel=False)

batch_size = 256
init_lr = 1e-3
log_iters = (train_imgs.shape[0] // batch_size) // 2
# print(train_imgs.shape, train_labs.shape, valid_imgs.shape, valid_labs.shape)

# normalize from [0, 255] to [0, 1]
train_imgs = (train_imgs - train_imgs.min()) / (train_imgs.max() - train_imgs.min())
valid_imgs = (valid_imgs - valid_imgs.min()) / (valid_imgs.max() - valid_imgs.min())

def model_choice(model_name):
    if model_name == 'MLP':
        model = nn.models.Model_MLP(
                [train_imgs.shape[-1], 256, 10],
                'ReLU',
                #[1e-3, 1e-3]
                )
        
    elif model_name == 'CNN':
        model = nn.models.Model_CNN(
            input_shape=(batch_size, 1, 28, 28),
            channel_list=[1, 4, 8, 16],
            kernel_size_list=[5, 5, 3],
            stride_list=[2, 2, 2],
            padding_list=[1, 1, 1],
            conv_func='ReLU',
            linear_out_list=[128, 10],
            linear_func='ReLU',
            #lambda_list_conv=[0.01, 0.01, 0.01],
            #lambda_list_linear=[0.01, 0.01],
        )
    else:
        raise ValueError('Model name not recognized!')
    return model

model = model_choice('CNN')

optimizer = nn.optimizer.SGD(init_lr=init_lr, model=model)
scheduler = None # nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999, lowest_lr=1e-4)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)#, l2=True, l2_lambda=1e-3)
#loss_fn = nn.op.MSELoss(model=model, one_hot=True, max_classes=train_labs.max()+1)
runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=batch_size, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=10, log_iters=log_iters, save_dir=r'./best_models_cnn_basic')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()