import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from rl_electric.environment_RL_load import CircuitEnv
from rl_electric.DQN_model import DQNAgent
from rl_electric.PG_model import PolicyGradientAgent


def main(controller):
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(23)

    n_steps = 1000
    dt = 0.001

    env = CircuitEnv(dt, n_steps)
    state = env.reset()
    state_size = len(state)

    if controller == "DQN":
        episodes = 1000
        action_size = 11
        agent = DQNAgent(state_size, action_size)
    elif controller == "PG":
        episodes = 10000
        action_size = 11
        agent = PolicyGradientAgent(state_size, action_size, continuous=False)
    elif controller == "PGcont":
        episodes = 10000
        agent = PolicyGradientAgent(state_size, 1, continuous=True)
    else:
        raise Exception("no valid controller")

    collected_rewards = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        rewards_episode = 0

        for _ in range(1, n_steps // 10):
            if controller in ["DQN", "PG"]:
                action_node = agent.select_action(state)
                action = action_node / (action_size - 1) * env.U_max - env.U_base
            elif controller == "PGcont":
                action = agent.select_action(state)
                action = np.clip(action, env.U_min - env.U_base, env.U_max - env.U_base)

            rewards = np.float32(0)
            for _ in range(10):
                next_state, reward = env.step(action)
                rewards += reward
            next_state = np.reshape(next_state, [1, state_size])

            if controller == "DQN":
                agent.remember(state.astype(np.float32), action_node, rewards.astype(np.float32), next_state.astype(np.float32))
                rewards_episode += rewards
            elif controller in ["PG", "PGcont"]:
                agent.store_reward(rewards)

            state = next_state

            if controller == "DQN":
                agent.learn()

                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * agent.tau + target_net_state_dict[key] * (1 - agent.tau)
                agent.target_net.load_state_dict(target_net_state_dict)

        if controller == "DQN":
            collected_rewards.append(rewards_episode)
        elif controller in ["PG", "PGcont"]:
            collected_rewards.append(np.sum(agent.rewards))
            agent.learn()
        print("Episode: ", e + 1)
        print("Return: ", collected_rewards[-1])

    agent.save(controller + "_model.pth")

    # plot losses
    if controller == "DQN":
        plt.plot(range(len(agent.collected_loss)), pd.Series(agent.collected_loss).ewm(alpha=0.1).mean())
        plt.yscale('log')
    elif controller in ["PG", "PGcont"]:
        plt.plot(range(len(agent.collected_loss)), agent.collected_loss)
    plt.xlabel('Iterationen')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig("loss_{}.png".format(controller))
    plt.clf()

    # plot returns
    plt.plot(range(episodes), collected_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Returns')
    plt.savefig("returns_{}.png".format(controller))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("controller")
    args = parser.parse_args()
    main(args.controller)
