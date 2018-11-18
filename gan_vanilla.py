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



# Training
# Reset grad

def reset_grad():
    for p in params:
        if p.grad is None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as(data).zero_())

G_solver = optim.Adam(G_params, lr=1e-3)
D_solver = optim.Adam(D_params, lr=1e-3)

ones_label = Variable(torch.ones(mb_size, 1))
zeros_labels = Variable(torch.zeros(mb_size, 1))

for it in range(100000):

    # Sample data

    z = Variable(torch.randn(mb_size, Z_dim))
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    G_sample = G(z)
    D_real =   D(X)
    D_fake = D(G_sample)

    D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
    D_loss_fake = nn.binary_cross_entropy(D_fake, zero_label)
    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()
    D_solver.step()


    ## Reset grad


    if it % 1000 == 0:
        print('Iter-{}; D-loss: {}; G-loss: {}'.format(it, D_loss.data_numpy(), G_loss.data.numpy()))

        samples = G(z).data.numpy()[:16]


        fig = plt.figure(figsize = (4, 4))
        gs = gridspec.GridSpec(4,4)
        gs.update(wspace = 0.05, hspace = 0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(samples.reshape(28, 28), cmap = 'Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c+=1
        plt.close(fig)


