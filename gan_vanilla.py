import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../MNIST_data', one_hot = True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1] # Shape of images
y_dim = mnist.train.labels.shape[1] # Shape of labels
h_dim = 128 # Dimensions
c = 0       # Cost function
lr = 1e-3   # Learning rate


# Xavier initialization
# Info
# http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization


# Xavier initialization of size
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn( * size) * xavier_stddev, requires_grad = True)

# Generator function
# Xavier initialization
# bias - h_dim zeros tensor

Wzh = xavier_init(size = [Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size = [h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad = True)

def G(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


# Discriminator function


Wxh = xavier_init(size = [X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad =True)


Why = xavier_init(size = [h_dim, 1])
bhy =  Variable(torch.zeros(1), requires_grad = True)

def D(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    y = nn.sigmoid(h @ Why + bhy.repeat(h.size(0), 1))
    return y





G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]


params = G_params + D_params

