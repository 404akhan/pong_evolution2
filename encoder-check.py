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

nn = pickle.load(open('encoder-weights-2.p', 'rb'))

def nn_forward(nn, inputs):
    hl1 = np.matmul(inputs, nn['W1'])
    hl1 = sigmoid(hl1)
    hl2 = np.matmul(hl1, nn['W2'])
    hl2 = sigmoid(hl2)

    return hl2

if reload:
    aver_loss = None
    for i_episode in xrange(1000*1000):
        images = []
        f(model, False)

        print 'encoded'
        index_to_show = np.random.randint(len(images))
        print images[index_to_show][images[index_to_show] != 0]
        plt.imshow(images[index_to_show].reshape(80, 80), cmap='gray')
        plt.show()

        np_images = np.array(images)
        images_decoded = nn_forward(nn, np_images)

        print 'decoded'
        print images_decoded[index_to_show][images_decoded[index_to_show] != 0]
        plt.imshow(images_decoded[index_to_show].reshape(80, 80), cmap='gray')
        plt.show()

    sys.exit('demo finished')
