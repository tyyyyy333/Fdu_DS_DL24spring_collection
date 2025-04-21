from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
        
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]
                    #print(f'layer param shape = {layer.params[key].shape}')
                    #print(f'layer param grad sum = {np.sum(self.init_lr * layer.grads[key])}')
                    #print(f'layer param grad mean = {np.mean(self.init_lr * layer.grads[key])}')
                    #print(self.init_lr * layer.grads[key])

    def clear_grad(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.grads.keys():
                    layer.clear_grad()


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.v = {}
        for layer in self.model.layers:
            if layer.optimizable == True:
                self.v[layer.layer_name] = {key: np.zeros_like(value, dtype=np.float32) for key, value in layer.params.items()}
                
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.v[layer.layer_name][key] = self.mu * self.v[layer.layer_name][key] + (1 - self.mu) * layer.grads[key]
                    layer.params[key] -= self.init_lr * self.v[layer.layer_name][key]
    def clear_grad(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.grads.keys():
                    layer.clear_grad()
                    
class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {}
        self.v = {}
        for layer in self.model.layers:
            if layer.optimizable == True:
                self.m[layer.layer_name] = {key: np.zeros_like(value) for key, value in layer.params.items()}
                self.v[layer.layer_name] = {key: np.zeros_like(value) for key, value in layer.params.items()}
                
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.m[layer.layer_name][key] = self.beta1 * self.m[layer.layer_name][key] + (1 - self.beta1) * layer.grads[key]
                    self.v[layer.layer_name][key] = self.beta2 * self.v[layer.layer_name][key] + (1 - self.beta2) * (layer.grads[key]**2)
                    m_hat = self.m[layer.layer_name][key] / (1 - self.beta1)
                    v_hat = self.v[layer.layer_name][key] / (1 - self.beta2)
                    layer.params[key] -= self.init_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    
    def clear_grad(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.grads.keys():
                    layer.clear_grad()