import math

import numpy as np


def noise(base_val, fraction, jump_lengths, n_steps):
    jumps = np.round(np.array(jump_lengths) * n_steps)
    jumps = np.append(jumps, n_steps - np.sum(jumps))

    noise = np.random.normal(0, fraction * base_val, len(jumps))

    noise_over_time = []
    for i, j in enumerate(jumps):
        noise_over_time += int(j) * [noise[i]]

    return noise_over_time


class CircuitEnv:
    def __init__(self, dt, n_steps):
        self.dt = dt
        self.n_steps = n_steps

        self.R_base = 5  # Ohm
        self.L_base = 0.5  # H
        self.U_base = 12  # V
        self.U_max = 15
        self.U_min = 0
        self.I_ref = 12 / self.R_base  # A

        self.R_t = noise(self.R_base, 0.15, [0.2, 0.4, 0.2], n_steps)
        self.L_t = noise(self.L_base, 0.25, [0.3, 0.2, 0.4], n_steps)

        self.state = None
        self.reset()

    def reset(self):
        self.steps = 0
        self.t = 0
        self.I = 0

        # self.state = [self.I, self.U_base]
        self.state = [self.I - self.I_ref, 0]
        return self.state

    def step(self, action):
        self.steps += 1
        self.t += self.dt

        I_last = self.I

        R = self.R_base + self.R_t[self.steps]
        L = self.L_base + self.L_t[self.steps]

        U = action + self.U_base

        # differential equation: dI/dt = (U - R * I) / L
        dI_dt = (U - R * I_last) / L
        self.I += dI_dt * self.dt

        self.I = max(0, self.I)

        # self.state = [self.I, U]
        self.state = [self.I - self.I_ref, action]

        reward = -np.abs(self.I - self.I_ref)

        return self.state, reward
