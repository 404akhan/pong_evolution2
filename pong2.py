# ES Pong

import gym
import numpy as np
import cPickle as pickle
import sys
import matplotlib.pyplot as plt

env = gym.make('Pong-v0')
np.random.seed(10)

input_dim = 80 * 80
hl_size = 200
version = 2
npop = 50
sigma = 0.1
alpha = 0.01
aver_reward = None
allow_writing = True
reload = True

print hl_size, version, npop, sigma, alpha

if reload:
    model = pickle.load(open('model-pong%d.p' % version, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(input_dim, hl_size) / np.sqrt(input_dim)
    model['W2'] = np.random.randn(hl_size, 1) / np.sqrt(hl_size)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def get_action(x, model):
    hl = np.matmul(x, model['W1'])
    hl = relu(hl)
    logp = np.matmul(hl, model['W2'])
    prob, = sigmoid(logp)
    action = 2 if np.random.uniform() < prob else 3
    return action

images = []

def f(model, render=False):
    global images
    state = env.reset()
    total_reward = 0
    prev_x = None
    for t in xrange(1000000):
        if render: env.render()

        cur_x = prepro(state)
        images.append(cur_x)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        prev_x = cur_x

        action = get_action(x, model)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if reward != 0: done = True
        if done:
            break
    return total_reward

nn = {}
param_size = 100
nn['W1'] = np.random.randn(6400, param_size) / np.sqrt(6400)
nn['W2'] = np.random.randn(param_size, 6400) / np.sqrt(param_size)
nn = pickle.load(open('encoder-weights-2.p', 'rb'))

nn_grad = {}
nn_grad_sq = {}
for k, v in nn.iteritems(): nn_grad[k] = np.zeros_like(v)
for k, v in nn.iteritems(): nn_grad_sq[k] = np.zeros_like(v)

def train_nn(nn, inputs, labels, lr=0.00003):
    # inputs, labels - np.array | bsize * 6400
    hl1 = np.matmul(inputs, nn['W1'])
    hl1 = sigmoid(hl1)
    hl2 = np.matmul(hl1, nn['W2'])
    hl2 = sigmoid(hl2)

    dhl2 = (hl2 - labels) / len(inputs)
    dhl2 *= hl2*(1 - hl2)
    dhl1 = np.matmul(dhl2, nn['W2'].transpose())
    dhl1 *= hl1*(1 - hl1)

    d = {}
    d['W2'] = np.matmul(hl1.transpose(), dhl2)
    d['W1'] = np.matmul(inputs.transpose(), dhl1)

    for k in nn_grad: nn_grad[k] = nn_grad[k] * 0.9 + d[k] * 0.1
    for k in nn_grad_sq: nn_grad_sq[k] = nn_grad_sq[k] * 0.999 + (d[k]**2) * 0.001
    for k in nn: nn[k] -= lr * nn_grad[k] / (np.sqrt(nn_grad_sq[k]) + 1e-5)

    return np.mean(np.square(hl2 - labels))

if reload:
    aver_loss = None
    for i_episode in xrange(1000*1000):
        # plt.imshow(images[-1].reshape(80, 80), cmap='gray')
        # plt.show()
        images = []
        f(model, False)
        np_images = np.array(images)
        cur_loss = train_nn(nn, np_images, np_images)

        aver_loss = aver_loss * 0.9 + cur_loss * 0.1 if aver_loss is not None else cur_loss
        print 'iter %d, cur_loss %f, aver_loss %f,' % (i_episode, cur_loss, aver_loss)

        if i_episode%100 == 0:
            pickle.dump(nn, open('encoder-weights-2.p', 'wb'))

    sys.exit('demo finished')
