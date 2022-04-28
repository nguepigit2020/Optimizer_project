import numpy as np
from config import args


# class optimizer:
#     def __init__(self,gamma,learning_rate,eps):
gamma = args.gamma
learning_rate = args.lr
eps = args.eps

def adagrad(g,G):
    G += g**2                                     # Update cache.
    step = learning_rate / (np.sqrt(G + eps)) * g
    return step

def momentum(grad,velocity):
    velocity = gamma* velocity + learning_rate * grad
    return velocity       

def rmsprop(grad,expected_grad):
    RMS_grad = np.sqrt(expected_grad + eps)
    step = (learning_rate/RMS_grad)*grad  
    return step
        
def adam(grad,mhat,vhat):
    step = 1 * learning_rate * mhat/(np.sqrt(vhat) + eps)
    return step 