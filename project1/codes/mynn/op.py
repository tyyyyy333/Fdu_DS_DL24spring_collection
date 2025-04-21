from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8, layer_name=None) -> None:
        super().__init__()
        self.shape = (in_dim, out_dim)
        self.W = initialize_method(loc=0, scale=0.5,size=(in_dim, out_dim))
        self.b = initialize_method(loc=0, scale=0.5,size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
        self.layer_name = layer_name    
        
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = np.dot(X, self.W) + self.b
        
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad) / self.input.shape[0]  #take out it if loss is nan
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / self.input.shape[0] #take out it if loss is nan
        #print(self.input.T.shape, grad.shape)
        #print('--------------------------------------')
        out = np.dot(grad, self.W.T)
        
        return out
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, input_shape, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8, layer_name=None) -> None:
        '''
        input_shape: [batch, channels, H, W]
        '''
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initialize_method = initialize_method

        self.W_ = self.initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b_ = self.initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W' : None, 'b' : None}
        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {'W' : self.W_, 'b' : self.b_}
        
        self.layer_name = layer_name
        
        self.input_shape = list(input_shape)
        self.input_shape[2] = input_shape[2] + 2 * padding
        self.input_shape[3] = input_shape[3] + 2 * padding
        
        self.output_shape = self.forward(np.random.randn(*input_shape)).shape
        
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.input = X
        
        new_h = int((X.shape[2] - self.kernel_size) / self.stride) + 1
        new_w = int((X.shape[3] - self.kernel_size) / self.stride) + 1
        output = np.full(shape=(X.shape[0],
                                self.out_channels,
                                new_h,
                                new_w
                                ),
                         fill_value=0
                         )
        for i in range(new_h):
            for j in range(new_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                X_window = X[:, :, h_start:h_end, w_start:w_end]
                
                for o in range(self.out_channels):
                    output[:, o, i, j] = np.sum(X_window * self.W_[o], axis=(1, 2, 3)) + self.b_[0, o, 0, 0]
        
        return output
        
    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size = grads.shape[0]
        X = self.input

        self.grads['W'] = np.zeros_like(self.W_)
        self.grads['b'] = np.zeros_like(self.b_)
        dX = np.zeros_like(X, dtype=np.float64)

        for i in range(grads.shape[2]):
            for j in range(grads.shape[3]):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                X_window = X[:, :, h_start:h_end, w_start:w_end]
                
                for o in range(grads.shape[1]):
                    delta = grads[:, o, i, j][:, None, None, None]
                    self.grads['W'][o] += np.sum(delta * X_window, axis=0)
                    self.grads['b'][0, o, 0, 0] += np.sum(grads[:, o, i, j])
                    dX[:, :, h_start:h_end, w_start:w_end] += delta * self.W_[o]

        self.grads['W'] /= batch_size
        self.grads['b'] /= batch_size

        if self.padding > 0:
            dX = dX[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dX

                    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
  
class flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False
        self.layer_name = 'flatten'
        
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = X.reshape(X.shape[0], -1)
        
        return output

    def backward(self, grads):
        return grads.reshape(self.input.shape)

class Dropout(Layer):
    def __init__(self, rate = 0.5, layer_name = 'dropout') -> None:
        super().__init__()
        self.rate = rate
        self.input = None
        self.mask = None
        self.optimizable = False
        self.layer_name = layer_name
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X, training = True):
        self.input = X
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=X.shape) / (1 - self.rate)
            output = X * self.mask
        else:
            output = X
        
        return output
    def backward(self, grads):
        return grads * self.mask
    
      
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        #print(np.sum(self.input))
        output = np.where(self.input < 0, 0, grads)
        return output

class LeakyReLU(Layer):
    def __init__(self, alpha = 0.1):
        super().__init__()
        self.input = None
        self.alpha = alpha
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.input = X
        output = np.where(X < 0, self.alpha * X, X)
        
        return output
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, self.alpha * grads, grads)
        
        return output

class Sigmoid(Layer):
    """
    An activation layer of sigmoid.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = 1 / (1 + np.exp(-X))
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        y = self.forward(self.input)
        
        output = grads * y * (1 - y)
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10,l2 = False, l2_lambda=1e-4) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.grads = None
        self.has_softmax = True
        self.pred = None
        self.labels = None
        self.l2 = l2
        if self.l2:
            self.l2_layer = L2Regularization(self.model, lambda_=l2_lambda)
        
    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        
        self.labels = labels
        if self.has_softmax:
            self.pred = softmax(predicts)
        else:
            self.pred = predicts
        one_hot = np.eye(self.max_classes)[self.labels]
        eps = 1e-3
        loss = -np.mean(np.sum(one_hot * np.log(self.pred + eps), axis=1))
        if self.l2:
            loss = self.l2_layer(loss)
            
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        one_hot = np.eye(self.max_classes)[self.labels]
        self.grads = self.pred - one_hot            
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)
        if self.l2:
            for layer in self.model.layers:
                if layer.optimizable:
                    layer.grads['W'] += self.l2_layer.lambda_ * layer.params['W'] * 2
        
    def cancel_soft_max(self):
        self.has_softmax = False
        return self

class MSELoss(Layer):
    def __init__(self, model = None, one_hot = False, max_classes = 10, l2 = False) -> None:
        super().__init__()
        self.model = model
        self.grads = None
        self.has_softmax = True
        self.max_classes = max_classes
        self.one_hot = one_hot
        self.pred = None
        self.labels = None
        self.l2 = l2
        if self.l2:
            self.l2_layer = L2Regularization(self.model, lambda_=1e-3)

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        if self.one_hot:
            self.labels = np.eye(self.max_classes)[labels]
        else:
            self.labels = labels
            
        if self.has_softmax:
            self.pred = softmax(predicts)
        else:
            self.pred = predicts
        
        loss = np.mean((self.pred - self.labels)**2)
        if self.l2:
            loss = self.l2_layer(loss)
            
        return loss

    def backward(self):
        self.grads = 2 * (self.pred - self.labels)
        self.model.backward(self.grads)
        if self.l2:
            for layer in self.model.layers:
                layer.grads['W'] += self.l2_layer.lambda_ * layer.params['W'] * 2

class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, lambda_ = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.lambda_ = lambda_

    def __call__(self, loss):
        return self.forward(loss)

    def forward(self, loss):
        w_sum = 0
        for layer in self.model.layers:
            if layer.optimizable:
                w_sum += np.sum(layer.params['W']**2)
        loss += self.lambda_ * w_sum
        return loss
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition