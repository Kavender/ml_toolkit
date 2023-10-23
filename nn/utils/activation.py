import numpy as np


def sigmoid(x):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    x -- numpy array of any shape
    
    Returns:
    s -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    s = 1/(1+np.exp(-x))
    cache = x
    
    return s, cache


def relu(x):
    """
    Implement the RELU function.

    Arguments:
    x -- A scalar or numpy array of any size.

    Returns:
    s -- Post-activation parameter, of the same shape as x
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    s = np.maximum(0,x)
    assert(s.shape == x.shape)
    cache = x 
    return s, cache