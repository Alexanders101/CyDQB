from random import random, randrange
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "theano"
import keras
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop
from keras import backend as K
from theano import printing


class RingBuffer():
    "A ND ring buffer using numpy arrays"

    def __init__(self, shape, dtype):
        self.data = np.zeros(shape, dtype=dtype)
        self.index = 0
        self.size = self.data.shape[0]

    def append(self, x):
        "adds array x to ring buffer"
        self.index = (self.index + 1) % self.size
        self.data[self.index] = x

    def pop(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + 1) % self.size
        return self.data[idx]

    def __getitem__(self, key):
        return self.data[key]


def build_model(frame_size, number_of_actions, save_name):
    init = lambda shape, name: keras.initializations.normal(shape, scale=0.01, name=name)

    S = Input(shape=frame_size)

    h = Convolution2D(32, 8, 8, activation='relu', subsample=(2, 2), init=init, border_mode='same')(S)
    h = Convolution2D(64, 4, 4, activation='relu', subsample=(2, 2), init=init, border_mode='same')(h)
    h = Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1), init=init, border_mode='same')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', init=init)(h)

    V = Dense(number_of_actions, init=init)(h)
    model = Model(S, V)
    try:
        model.load_weights('{}.h5'.format(save_name))
        print("loading from {}.h5".format(save_name))
    except:
        print("Training a new model")

    return model


class DQNModel:
    def __init__(self, model, optimizer, gamma):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma

        # self.create_functions()
        self.model.compile(optimizer, 'mse')

        NS = Input(shape=self.model.input_shape[1:], dtype='float32')
        self.grad_model = K.function([NS], [K.stop_gradient(self.model(NS))])

    def create_functions(self):
        input_shape = self.model.input_shape[1:]

        S = Input(shape=input_shape, dtype='float32')
        NS = Input(shape=input_shape, dtype='float32')
        A = K.placeholder(shape=(None,), dtype='int16')
        R = K.placeholder(shape=(None,), dtype='float32')
        T = K.placeholder(shape=(None,), dtype='uint8')

        self.predict_ = K.function([S], self.model(S))

        VS = self.model(S)
        VNS = K.stop_gradient(self.model(NS))

        future_value = (1 - T) * K.max(VNS, axis=1)
        target = R + (self.gamma * future_value)
        cost = K.mean(((VS[:, A] - target) ** 2))
        updates = self.optimizer.get_updates(self.model.trainable_weights, [], cost)
        self.train_ = K.function([S, NS, A, R, T], cost, updates=updates)
        x_printed = printing.Print('VS[:, A]')(VS[:, A])

        self.debug = K.function([S, NS, A, R, T], x_printed)

    def predict(self, X):
        # return self.predict_((X,))
        return self.model.predict(X)

    def fit(self, S, NS, A, R, T):
        # return self.train_((S, NS, A, R, T))
        VS = self.model.predict(S)
        VNS = self.grad_model((NS.astype(np.float32),))[0]

        target = R + (self.gamma * (1 - T) * VNS.max(axis=1))
        diff = target - VS[np.arange(VS.shape[0]), A]
        return self.model.train_on_batch(S, (VS.T + diff).T)

    def save_weights(self, filepath, overwrite=True):
        self.model.save_weights(filepath, overwrite)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)


class Agent(object):
    def __init__(self, game, frame_size, num_actions, save_name, epsilon=0.1, delta_epsilon=0.00001, gamma=0.99,
                 batch_size=64, frame_seq_count=1, memory=5000, save_freq=10, optimizer=RMSprop(0.0001)):
        self.game = game

        self.channel_axis = 2 if K.image_dim_ordering() == "tf" else 0
        self.colors = frame_size[self.channel_axis]

        self.frame_size = frame_size
        self.state_size = list(frame_size)
        self.state_size[self.channel_axis] = frame_seq_count * self.colors
        self.state_size = tuple(self.state_size)

        self.num_actions = num_actions
        self.frame_seq_count = frame_seq_count

        self.epsilon = epsilon
        self.delta_epsilon = delta_epsilon
        self.gamma = gamma

        self.save_freq = save_freq
        self.memory = memory
        self.batch_size = batch_size

        self.save_name = save_name

        print("Compiling Model...")
        self.model = DQNModel(build_model(self.state_size, num_actions, save_name), optimizer, gamma)
        print("Done")

        print("Starting Agent with the following parameters:")
        print("epsilon: {}".format(epsilon))
        print("delta_epsilon: {}".format(delta_epsilon))
        print("batch_size: {}".format(batch_size))
        print("frame_seq_count: {}".format(frame_seq_count))
        print("memory: {}".format(memory))
        print("optimizer: {}".format(optimizer))

    def play(self, num_episodes, train=True):
        D_S = RingBuffer((self.memory,) + self.state_size, np.uint8)
        D_NS = RingBuffer((self.memory,) + self.state_size, np.uint8)
        D_A = RingBuffer((self.memory,), np.uint16)
        D_R = RingBuffer((self.memory,), np.float32)
        D_T = RingBuffer((self.memory,), np.uint8)
        batch_choices = np.arange(self.memory)
        frame_size = (1,) + self.frame_size

        # Setup the initial state of the game
        x_0 = self.game.reset().reshape(frame_size)
        s_0 = np.repeat(x_0, self.frame_seq_count, axis=self.channel_axis + 1)

        t = 0
        for episode in range(num_episodes):
            print("====================")
            print("Episode {} has begun".format(episode))
            self.game.reset()
            s_t = s_0
            terminal = 0

            total_loss = 0
            total_score = 0

            if self.epsilon > 0 and t > self.memory:
                self.epsilon -= self.delta_epsilon

            while not terminal:
                # Get the Models chosen Action or random action
                values = self.model.predict(s_t)
                a_t = (randrange(self.num_actions)
                       if random() <= self.epsilon
                       else values.argmax())

                x_t1, r_t, terminal = self.game.step(a_t, values)
                x_t1 = x_t1.reshape(frame_size)
                s_t1 = np.concatenate((s_t[:, :, :, self.colors:], x_t1), axis=self.channel_axis + 1)

                total_score += r_t
                D_S.append(s_t[0])
                D_NS.append(s_t1[0])
                D_A.append(a_t)
                D_R.append(r_t)
                D_T.append(terminal)

                if t > self.memory and train:
                    batch = np.random.choice(batch_choices, self.batch_size, replace=False)
                    total_loss += self.model.fit(D_S[batch], D_NS[batch], D_A[batch], D_R[batch], D_T[batch])

                s_t = s_t1
                t += 1

            print("Total Loss: {:.4f} | Total Score: {:.4f} | Epsilon: {:.5f} | Frames: {}".format(total_loss,
                                                                                                   total_score,
                                                                                                   self.epsilon, t))
            if episode % self.save_freq == 0:
                print("Saving Model to {}.j5".format(self.save_name))
                self.model.save_weights("{}.h5".format(self.save_name))
