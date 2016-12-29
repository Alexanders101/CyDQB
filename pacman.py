import gym
from c_FastDQN import Agent
import keras
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop
from keras import backend as K

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

class MsPacMan:
    def __init__(self, render=True):
        self.env = gym.make("MsPacman-v0")
        self.num_actions = self.env.action_space.n
        self.frame_size = self.env.observation_space.shape
        self.render = render

    def step(self, action, value):
        if self.render:
            self.env.render()

        observation, reward, done, info = self.env.step(action)
        return observation, reward, done

    def reset(self):
        return self.env.reset()


if __name__ == '__main__':
    game = MsPacMan(True)
    agent = Agent(build_model, game, game.frame_size, game.num_actions, "MsPacMan", frame_seq_count=3, save_freq=5, memory=100)
    agent.play(10000)