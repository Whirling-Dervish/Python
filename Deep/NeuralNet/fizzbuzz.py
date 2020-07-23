"""
For each of the numbers from 1 to 100 :
* If the number is divisible by 3, print "fizz"
* If the number is divisible by 5, print "buzz"
* If the number is divisible by 3, print "fizzbuzz"
* Otherwise print the number itself
"""

import numpy as np
from typing import List
from geenet.train import train
from geenet.nn import NeuralNet
from geenet.layers import Linear, Tanh
from geenet.optim import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0,0,0,1]
    elif x % 5 == 0:
        return [0,0,1,0]
    elif x % 3 == 0:
        return [0,1,0,0]
    else:
        return [1,0,0,0]

def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x

    Args:
        x (int): [description]

    Returns:
        List[int]: 10 digit binary encoding of a numbers
                   5 -> [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                   6 -> [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]


    """
    return [x >> i & 1 for i in range(10)]

inputs = np.array(
    [ binary_encode(x) for x in range(101,1024)]
    )

targets = np.array(
    [fizz_buzz_encode(x) for x in range(101,1024)]
    )

net = NeuralNet([
    Linear(input_size = 10, output_size =50),
    Tanh(),
    Linear(input_size = 50, output_size = 4)
])

train(net, 
    inputs, 
    targets, 
    num_epochs = 500,
    optimizer = SGD(lr=0.0001))

for x in range(1,101):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x),"fizz","buzz", "fizzbuzz"]
    print(x, labels[predicted_idx],labels[actual_idx])
