# ES Pong

import gym
import numpy as np
import cPickle as pickle

env = gym.make('Pong-v0')
np.random.seed(10)

input_dim = 100 # 80 * 80
hl_size = 50
version = 3
npop = 50
sigma = 0.1
alpha = 0.003
aver_reward = None
aver_pop = None
allow_writing = True
reload = True

print hl_size, version, npop, sigma, alpha

nn = pickle.load(open('encoder-weights-2.p', 'rb'))
if reload:
    print 'loading weights from', 'model-pong%d.p' % version
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

def convert(x):
    hl1 = np.matmul(x, nn['W1'])
    hl1 = sigmoid(hl1)

    return hl1

def f(model, render=False):
    state = env.reset()
    total_reward = 0
    for t in xrange(1000000):
        if render: env.render()

        cur_x = prepro(state)
        x = convert(cur_x)

        action = get_action(x, model)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if reward != 0: done = True
        if done:
            break
    return total_reward

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
    if i % 10 == 0 and allow_writing:
        pickle.dump(model, open('model-pong%d.p' % version, 'wb'))
    cur_reward = f(model)
    mean_pop = np.mean(R)
    aver_reward = aver_reward * 0.9 + cur_reward * 0.1 if aver_reward != None else cur_reward
    aver_pop = aver_pop * 0.9 + mean_pop * 0.1 if aver_pop != None else mean_pop
    print('iter %d, mean_pop %.2f, aver_pop %.2f, cur_reward %.2f, aver_reward %.2f' % (i, mean_pop, aver_pop, cur_reward, aver_reward))


