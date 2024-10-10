import argparse

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

import torch

from rl_electric.environment_RL_load import CircuitEnv
from rl_electric.DQN_model import DQNAgent
from rl_electric.PG_model import PolicyGradientAgent
from rl_electric.MPC_model import MPCController

from IPython import embed


def main(controller):
    np.random.seed(42)
    torch.manual_seed(23)

    n_steps = 1000
    dt = 0.001
    t_steps = np.linspace(0, n_steps*dt, n_steps)

    env = CircuitEnv(dt, n_steps)
    state = env.reset()
    state_size = len(state)

    action_size = 11

    Irefs = np.zeros(n_steps)
    Is = np.zeros(n_steps)
    Us = np.zeros(n_steps)
    returns = 0

    delta_Is = []

    if controller == "DQN":
        agent = DQNAgent(state_size, action_size)
        agent.load("DQN_model.pth")
    elif controller == "PG":
        agent = PolicyGradientAgent(state_size, action_size, continuous=False)
        agent.load("PG_model.pth")
    elif controller == "PGcont":
        agent = PolicyGradientAgent(state_size, 1, continuous=True)
        agent.load("PGcont_model.pth")

    for i in range(1, n_steps):
        if i % 10 - 1 == 0:
            if controller == "PID":
                delta_Is.append(state[0])
                if len(delta_Is) < 2:
                    gradient_error = 0
                else:
                    gradient_error = np.gradient(delta_Is)[-1]
                integral_error = cumulative_trapezoid(delta_Is, dx=env.dt, initial=0)[-1]
                K_p = -70
                K_i = -50
                K_d = -5
                action = K_p * delta_Is[-1] + K_i * integral_error + K_d * gradient_error
                action = np.clip(action, env.U_min - env.U_base, env.U_max - env.U_base)

            elif controller in ["DQN", "PG"]:
                action_node = agent.select_action(np.reshape(state, [1, state_size]), inference=True)
                action = action_node / (action_size - 1) * env.U_max - env.U_base
    
            elif controller == "PGcont":
                action = agent.select_action(np.reshape(state, [1, state_size]), inference=True).detach().cpu().item()
                action = np.clip(action, env.U_min - env.U_base, env.U_max - env.U_base)

            elif controller == "MPC":
                agent = MPCController(env)
                action = agent.optimize_voltage([state[0] + env.I_ref, state[1] + env.U_base]) - env.U_base

            else:
                action = 0

        next_state, reward = env.step(action)

        returns += reward
        Irefs[i] = env.I_ref
        Is[i] = state[0] + env.I_ref
        Us[i] = state[1] + env.U_base

        state = next_state

    print("return: ", returns)

    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(t_steps, Is, label='Istwert Stromstärke')
    ax1.plot(t_steps, Irefs, label='Sollwert Stromstärke')
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel('Stromstärke [A]')
    ax2 = ax1.twinx()
    ax2.plot(t_steps, Us, label='Spannung', color="black")
    ax2.set_ylabel('Spannung [V]')
    ax2.axhline(y=env.U_max, color="black", linestyle="dashed")
    ax2.axhline(y=env.U_base, color="black", linestyle="dashed")
    plt.axvline(x=0.2, color="grey", linestyle="dashed")
    plt.axvline(x=0.3, color="grey", linestyle="dashed")
    plt.axvline(x=0.5, color="grey", linestyle="dashed")
    plt.axvline(x=0.6, color="grey", linestyle="dashed")
    plt.axvline(x=0.8, color="grey", linestyle="dashed")
    plt.axvline(x=0.9, color="grey", linestyle="dashed")
    plt.title('Stromverlauf in einem RL-Schaltkreis')
    fig.legend()
    plt.show()

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", nargs='?', const="no")
    args = parser.parse_args()
    main(args.controller)
