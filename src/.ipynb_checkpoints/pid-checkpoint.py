import numpy as np

class PID:
    def __init__(self):
        self.e_t = []

        self.Kp = 1e-3
        self.Ki = 1e-3

        self.last_action = 0

    def __call__(self, set_theta, theta):
        self.e_t.append(set_theta - theta)
        err = self.Kp * self.e_t[-1] + \
            self.Ki * sum([ e for e in self.e_t ]) * len(self.e_t) + \
            (self.e_t[-1] - self.e_t[-2]) / len(self.e_t) if len(self.e_t) > 1 else 0
        self.Kp += err
        self.Ki += err
        return 0 if err > 0 else 1

    def update(self, last_action):
        self.last_action = last_action
