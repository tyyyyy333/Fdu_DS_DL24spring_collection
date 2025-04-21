from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func
        self.training = True
        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                layer.layer_name = f'Linear_{i}'
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Sigmoid':
                    layer_f = Sigmoid()
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                elif act_func == 'LeakyReLU':
                    layer_f = LeakyReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                #layer_d = Dropout(0.1)
                #self.layers.append(layer_d)
        #layer_out = Logistic()
        #self.layers.append(layer_out)
        
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            if self.training and isinstance(layer, Dropout):
                outputs = layer.forward(outputs,training=self.training)
            else:
                outputs = layer.forward(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)   
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.layer_name = f'Linear_{i}'
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Sigmoid':
                    layer_f = Sigmoid()
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        
class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self,
                 input_shape = None,
                 channel_list = None,
                 kernel_size_list = None,
                 stride_list = None,
                 padding_list = None,
                 conv_func = None,
                 linear_out_list = None,
                 linear_func = None,
                 lambda_list_conv = None,
                 lambda_list_linear = None
                 ):
        self.input_shape = input_shape
        self.flowing_shape = input_shape
        self.channel_list = channel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.conv_func = conv_func
        self.lambda_list_conv = lambda_list_conv
        
        self.linear_out_list = linear_out_list
        self.linear_func = linear_func
        self.lambda_list_linear = lambda_list_linear

        self.layers = []
        if self.channel_list is not None:
            for i in range(len(self.channel_list) - 1):
                layer = conv2D(
                    input_shape = self.flowing_shape,
                    in_channels=self.channel_list[i],
                    out_channels=self.channel_list[i + 1],
                    kernel_size=self.kernel_size_list[i],
                    stride=self.stride_list[i],
                    padding=self.padding_list[i]
                    )
                layer.layer_name = f'Conv2D_{i}'
                
                self.layers.append(layer)
                self.flowing_shape = layer.output_shape
                
                if self.lambda_list_conv is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = self.lambda_list_conv[i]
                    
                if self.conv_func == 'ReLU':
                    layer_f = ReLU()
                elif self.conv_func == 'LeakyReLU':
                    layer_f = LeakyReLU()
                elif self.conv_func == 'Sigmoid':
                    layer_f = Sigmoid()
                self.layers.append(layer_f)
            
            self.layers.append(flatten())
            
            assert len(self.flowing_shape) == 4, "Flattened shape should be 4D"
            self.flowing_shape = self.flowing_shape[1] * self.flowing_shape[2] * self.flowing_shape[3]
            
        if self.linear_out_list is not None:    
            for i in range(len(self.linear_out_list)):
                layer = Linear(in_dim=self.flowing_shape, out_dim=self.linear_out_list[i])
                layer.layer_name = f'Linear_{i}'
                self.layers.append(layer)
                self.flowing_shape = self.linear_out_list[i]
                if self.lambda_list_linear is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = self.lambda_list_linear[i]
                
                if self.linear_func == 'ReLU':
                    layer_f = ReLU()
                elif self.linear_func == 'LeakyReLU':
                    layer_f = LeakyReLU()
                elif self.linear_func == 'Sigmoid':
                    layer_f = Sigmoid()
                    
                if i < len(self.linear_out_list) - 1:
                    self.layers.append(layer_f)
        
        
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        (
            self.input_shape,
            self.channel_list,
            self.kernel_size_list,
            self.stride_list,
            self.padding_list,
            self.conv_func,
            self.lambda_list_conv,
            self.linear_out_list,
            self.linear_func,
            self.lambda_list_linear,
            saved_params
        ) = param_list
        
        self.flowing_shape = self.input_shape
        self.layers = []

        for i in range(len(self.channel_list) - 1):
            layer = conv2D(
                input_shape=self.flowing_shape,
                in_channels=self.channel_list[i],
                out_channels=self.channel_list[i + 1],
                kernel_size=self.kernel_size_list[i],
                stride=self.stride_list[i],
                padding=self.padding_list[i]
            )
            layer.layer_name = f'Conv2D_{i}'
            layer.params['W'] = saved_params[i]['W']
            layer.params['b'] = saved_params[i]['b']
            layer.weight_decay = saved_params[i]['weight_decay']
            layer.weight_decay_lambda = saved_params[i]['lambda']
            self.layers.append(layer)

            self.flowing_shape = layer.output_shape

            if self.conv_func == 'ReLU':
                self.layers.append(ReLU())
            elif self.conv_func == 'LeakyReLU':
                self.layers.append(LeakyReLU())
            elif self.conv_func == 'Sigmoid':
                self.layers.append(Sigmoid())

        self.layers.append(flatten())
        self.flowing_shape = self.flowing_shape[1] * self.flowing_shape[2] * self.flowing_shape[3]
        linear_start_idx = len(self.channel_list) - 1
        for i in range(len(self.linear_out_list)):
            layer = Linear(in_dim=self.flowing_shape, out_dim=self.linear_out_list[i])
            layer.layer_name = f'Linear_{i}'
            layer.params['W'] = saved_params[linear_start_idx + i]['W']
            layer.params['b'] = saved_params[linear_start_idx + i]['b']
            layer.weight_decay = saved_params[linear_start_idx + i]['weight_decay']
            layer.weight_decay_lambda = saved_params[linear_start_idx + i]['lambda']
            self.layers.append(layer)

            self.flowing_shape = self.linear_out_list[i]

            if i < len(self.linear_out_list) - 1:
                if self.linear_func == 'ReLU':
                    self.layers.append(ReLU())
                elif self.linear_func == 'LeakyReLU':
                    self.layers.append(LeakyReLU())
                elif self.linear_func == 'Sigmoid':
                    self.layers.append(Sigmoid())

    def save_model(self, save_path):
        saved_params = []

        for layer in self.layers:
            if hasattr(layer, 'params') and 'W' in layer.params:
                saved_params.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': getattr(layer, 'weight_decay', False),
                    'lambda': getattr(layer, 'weight_decay_lambda', 1e-8)
                })

        param_list = [
            self.input_shape,
            self.channel_list,
            self.kernel_size_list,
            self.stride_list,
            self.padding_list,
            self.conv_func,
            self.lambda_list_conv,
            self.linear_out_list,
            self.linear_func,
            self.lambda_list_linear,
            saved_params
        ]

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)