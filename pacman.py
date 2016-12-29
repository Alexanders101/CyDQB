import gym
from FastDQN import Agent

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
    agent = Agent(game, game.frame_size, game.num_actions, "MsPacMan", frame_seq_count=3, save_freq=5)
    agent.play(10000)