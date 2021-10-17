import gym
import matplotlib.pyplot as plt
import numpy as np

from pid import PID

# observations = [position of cart, velocity of cart, angle of pole, rotation rate of pole]
#0     Push cart to the left
#1     Push cart to the right

class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def run(self, pid):
        history = []

        observation = self.env.reset()
        is_done = False
        while not is_done:
            counter = 0
            while True:
                self.env.render()
                # Note: The set point is zero rotational angle theta and zero angular velocity theta_dot
                history.append([ 0.0, observation[2] ])
                action = pid(0.0, observation[2])
                observation, reward, done, info = self.env.step(action)
                pid.update(action)

                if done and observation[2] > np.pi / 6:
                    self.env.reset()
                    print('End of episodes')
                    break

                counter += 1

                if counter > 120:
                    print('Accomplished balance')
                    is_done = True
                    break

        return history

    def __del__(self):
        self.env.close()
