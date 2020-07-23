"""Loss functions measure how good our predictions are.
We can use them to adjust our parameters.
"""
import numpy as np
from geenet.tensor import Tensor

class Loss:
    def loss(self, predicted:Tensor, actual:Tensor) -> float :
        raise NotImplementedError

    def grad(self, predicted:Tensor, actual: Tensor) -> Tensor :
        # returns a tensor of partial derivatives
        raise NotImplementedError

class MSE(Loss):
    """The MSE in our case is actually total squared error
    """
    def loss(self, predicted:Tensor, actual:Tensor) -> float :
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted:Tensor, actual: Tensor) -> Tensor :
        # Gradient of squared loss is just two times the loss
        return 2 * (predicted - actual)



