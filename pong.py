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
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        images.append(x)
        prev_x = cur_x

        action = get_action(x, model)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if reward != 0: done = True
        if done:
            break
    return total_reward

nn = {}
param_size = 200
nn['W1'] = np.random.randn(6400, param_size) / np.sqrt(6400)
nn['W2'] = np.random.randn(param_size, 24) / np.sqrt(param_size)
nn['W3'] = np.random.randn(24, param_size) / np.sqrt(24)
nn['W4'] = np.random.randn(param_size, 6400) / np.sqrt(param_size)

nn_grad = {}
nn_grad_sq = {}
for k, v in nn.iteritems(): nn_grad[k] = np.zeros_like(v)
for k, v in nn.iteritems(): nn_grad_sq[k] = np.zeros_like(v)

def train_nn(nn, inputs, labels, lr=0.1):
    # inputs, labels - np.array | bsize * 6400
    hl1 = np.matmul(inputs, nn['W1'])
    hl1[hl1 <= 0] = 0
    hl2 = np.matmul(hl1, nn['W2'])
    hl2[hl2 <= 0] = 0
    hl3 = np.matmul(hl2, nn['W3'])
    hl3[hl3 <= 0] = 0
    hl4 = np.matmul(hl3, nn['W4'])
    hl4 = np.tanh(hl4)

    dhl4 = (hl4 - labels) / len(inputs)
    dhl4 *= (1 - hl4*hl4)
    dhl3 = np.matmul(dhl4, nn['W4'].transpose())
    dhl3[hl3 <= 0] = 0
    dhl2 = np.matmul(dhl3, nn['W3'].transpose())
    dhl2[hl2 <= 0] = 0
    dhl1 = np.matmul(dhl2, nn['W2'].transpose())
    dhl1[hl1 <= 0] = 0

    d = {}
    d['W4'] = np.matmul(hl3.transpose(), dhl4)
    d['W3'] = np.matmul(hl2.transpose(), dhl3)
    d['W2'] = np.matmul(hl1.transpose(), dhl2)
    d['W1'] = np.matmul(inputs.transpose(), dhl1)

    for k in nn_grad: nn_grad[k] = nn_grad[k] * 0.9 + d[k] * 0.1
    for k in nn_grad_sq: nn_grad_sq[k] = nn_grad_sq[k] * 0.999 + (d[k]**2) * 0.001
    for k in nn: nn[k] -= lr * nn_grad[k] / (np.sqrt(nn_grad_sq[k]) + 1e-5)

    return np.mean(np.square(hl4 - labels))

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

        if i_episode%10 == 0:
            pickle.dump(nn, open('encoder-weights.p', 'wb'))

    sys.exit('demo finished')

for i in xrange(100001):
    N = {}
    for k, v in model.iteritems(): N[k] = np.random.randn(npop, v.shape[0], v.shape[1])
    R = np.zeros(npop)
    for j in range(npop):
        model_try = {}
        for k, v in model.iteritems(): model_try[k] = v + sigma*N[k][j]
        R[j] = f(model_try)
    A = (R - np.mean(R)) / (np.std(R) + 1e-5)
    for k in model: model[k] = model[k] + alpha/(npop*sigma) * np.dot(N[k].transpose(1, 2, 0), A)
    if i % 1 == 0 and allow_writing:
        pickle.dump(model, open('model-pong%d.p' % version, 'wb'))
    cur_reward = f(model)
    aver_reward = aver_reward * 0.9 + cur_reward * 0.1 if aver_reward != None else cur_reward
    print('iter %d, mean_pop %.2f, cur_reward %.2f, aver_reward %.2f' % (i, np.mean(R), cur_reward, aver_reward))
