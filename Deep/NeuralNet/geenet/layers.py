""" Each layer needs to pass its inputs forward
and propagate gradients backwards 
"""
import numpy as np
from geenet.tensor import Tensor
from typing import Dict, Callable

class Layer:

    def __init__(self) -> None:           
        self.params: Dict[str,Tensor]= {}
        self.grads: Dict[str,Tensor]= {}
    
    def forward(self, inputs:Tensor) -> Tensor :
        """Produce the outputs corresponding to the inputs
        """
        raise NotImplementedError

    def backward(self, grad:Tensor) -> Tensor:
        """
        Back propagate the gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    and computes the output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size:int) -> None:
        super().__init__()
        self.params['w'] = np.random.randn(input_size,output_size)
        self.params['b'] = np.random.rand(output_size)

    def forward(self, inputs:Tensor) -> Tensor :
        """Forward pass

        Args:
            inputs (Tensor): inputs

        Returns:
            Tensor:  inputs @ w + backwards
        """
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad:Tensor) -> Tensor :
        """
        if y = f(x) and x = a * b + c
        dy/da = f'(x) * b 
        dy/db = f'(x) * a
        dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        dy/da = f'(x) @ b.T
        dy/db = a.T @ f'(x) 
        dy/dc  = f'(x)

        Args:
            grad (Tensor): [description]

        Returns:
            Tensor: [description]
        """
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T

F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    An activation layer just applies as function
    elementwise to its inputs.
    """
    def __init__(self, f: F, f_prime:F) -> None:
        super().__init__()
        self.f_prime = f_prime
        self.f = f
    
    def forward(self, inputs:Tensor) -> Tensor :
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad:Tensor) -> Tensor :
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
    
        Args:
            grad (Tensor): [description]

        Returns:
            Tensor: [description]
        """
        return self.f_prime(self.inputs) * grad

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)