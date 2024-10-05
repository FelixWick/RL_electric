from collections import deque
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# pip install torch==1.13 --index-url https://download.pytorch.org/whl/cu116
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print(device)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_size, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, action_size)
        )

    def forward(self, X):
        return self.mlp(X)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.replay_memory = deque(maxlen=10000)

        self.batch_size = 32

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.tau = 0.005

        self.policy_net = DQN(state_size, self.action_size).to(device)
        self.target_net = DQN(state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.criterion = nn.MSELoss()

        self.collected_loss = []

    def remember(self, state, action_node, reward, next_state):
        self.replay_memory.append((state, action_node, reward, next_state))

    def select_action(self, state, inference=False):
        if (np.random.rand() <= self.epsilon) and (inference == False):
            return random.randrange(self.action_size)
        state = torch.tensor(state.astype(np.float32)).to(device)
        with torch.no_grad():
            Q_values = self.policy_net(state)
        return torch.argmax(Q_values).item()

    def learn(self):
        if len(self.replay_memory) < self.batch_size:
            return
        minibatch = list(zip(*random.sample(self.replay_memory, self.batch_size)))
        state = torch.tensor(np.array(minibatch[0])).squeeze(2).to(device)
        action_node = torch.tensor(minibatch[1]).unsqueeze(1).to(device)
        reward = torch.tensor(np.array(minibatch[2])).to(device)
        next_state = torch.tensor(np.array(minibatch[3])).squeeze(2).to(device)

        output = self.policy_net(state)
        pred = torch.gather(output.squeeze(), 1, action_node).squeeze()

        target = reward + self.gamma * torch.max(self.target_net(next_state))

        loss = self.criterion(pred, target)
        loss.backward()
        self.collected_loss.append(loss.detach().cpu().item())

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        torch.save(self.policy_net.state_dict(), name)

    def load(self, name):
        self.policy_net.load_state_dict(torch.load(name))
