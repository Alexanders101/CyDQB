import game.wrapped_flappy_bird as FLAPPY
import numpy as np
from keras.optimizers import Adam
from skimage import transform, color, exposure

from FastDQN.py_FastDQN import Agent


class flappy_game:
    def __init__(self):
        self.game = FLAPPY.GameState()

    def transform_frame(self, frame):
        frame = color.rgb2gray(frame)
        frame = transform.resize(frame, (80, 80))
        frame = exposure.rescale_intensity(frame, out_range=(0, 255))
        frame = frame.round()
        frame = frame.astype(np.uint8)

        return frame

    def step(self, action, values):
        a = np.zeros(2, np.uint16)
        a[action] = 1

        x_t1, r_t, terminal = self.game.frame_step(a, values)
        x_t1 = self.transform_frame(x_t1)
        x_t1.resize((1, 80, 80, 1))

        return x_t1, r_t, terminal

    def reset(self):
        self.game.__init__()
        return self.step(0, None)[0]


if __name__ == '__main__':
    GAME = flappy_game()
    agent = Agent(GAME, (80,80,1), 2, 'flappy', frame_seq_count=4, optimizer=Adam(1e-6), memory=10000, delta_epsilon=0.1 / 2E6)
    agent.play(20000, 1)
