import gym
import keras
from FastDQN import Agent
from keras.layers import Convolution2D, Dense, Flatten, Input
from keras.models import Model
from skimage import transform, color, exposure, util
import warnings
warnings.filterwarnings("ignore")



def build_model(state_size, number_of_actions):

    S = Input(shape=state_size)

    h = Convolution2D(32, 8, 8, subsample=(2, 2), activation='relu', init='glorot_normal', border_mode='same')(S)
    h = Convolution2D(64, 4, 4, activation='relu', subsample=(2, 2), init='glorot_normal', border_mode='same')(h)
    h = Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1), init='glorot_normal', border_mode='same')(h)
    h = Flatten()(h)
    h = Dense(512, activation='relu', init='glorot_normal')(h)

    V = Dense(number_of_actions, init='glorot_normal')(h)
    model = Model(S, V)
    return model

class MsPacMan:
    def __init__(self, render=True):
        self.env = gym.make("MsPacman-v0")
        self.num_actions = self.env.action_space.n
        self.frame_size = self.env.observation_space.shape
        self.render = render

    def step(self, action, value=None):
        if self.render:
            self.env.render()
        # print(value)
        observation, reward, done, info = self.env.step(action)
        return self.transform(observation), reward, done

    def reset(self):
        return self.transform(self.env.reset())

    def transform(self, frame):
        return util.img_as_ubyte(color.rgb2gray(frame)[2:170, :]).reshape(168,160,1)

if __name__ == '__main__':
    game = MsPacMan(True)
    # from matplotlib.pyplot import imsave
    # imsave('testframe.png', game.reset().reshape(168,160))
    agent = Agent(build_model=build_model,
                  game=game,
                  frame_size=(168,160,1),
                  num_actions=game.num_actions,
                  save_name="MsPacMan",
                  frame_seq_count=4,
                  save_freq=5,
                  memory=5000,
                  epsilon=0.1,
                  delta_epsilon=0.005)

    agent.play(10000, True)
