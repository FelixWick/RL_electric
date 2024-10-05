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


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_size, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, X):
        return self.mlp(X)


class PolicyNetworkContinuous(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size, 20)
        self.fc2 = nn.Linear(20, 20)
        
        self.mean = nn.Linear(20, action_size)
        self.log_std = nn.Linear(20, action_size)

    def forward(self, X):
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))

        mean = self.mean(X)
        log_std = self.log_std(X)
        std = torch.exp(log_std)
        return mean, std


class PolicyGradientAgent:
    def __init__(self, state_size, action_size, continuous):
        self.continuous = continuous
        self.action_size = action_size

        self.gamma = 0.95

        if self.continuous:
            self.model = PolicyNetworkContinuous(state_size, self.action_size).to(device)
        else:
            self.model = PolicyNetwork(state_size, self.action_size).to(device)

        self.optimizer = optim.Adam(self.model.parameters())

        self.log_probs = []
        self.rewards = []

        self.collected_loss = []

    def store_reward(self, reward):
        self.rewards.append(reward)

    def select_action(self, state, inference=False):
        state = torch.FloatTensor(state).to(device)

        if self.continuous:
            mean, std = self.model(state)
            action = torch.normal(mean, std)

            action_dist = torch.distributions.Normal(mean, std)
            log_prob = action_dist.log_prob(action)
            self.log_probs.append(log_prob)

            if inference:
                return mean
            else:
                return action.item()

        else:
            probs = self.model(state)
            if inference:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()
            self.log_probs.append(torch.log(probs[:, action]))
            return action

    def learn(self):
        discounted_returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)

        discounted_returns = torch.FloatTensor(discounted_returns).to(device)
        discounted_returns -= discounted_returns.mean()

        loss = []
        for log_prob, reward in zip(self.log_probs, discounted_returns):
            loss.append(-log_prob * reward)  # minus sign: gradient ascent to increase rewards
        loss = torch.cat(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.collected_loss.append(loss.detach().cpu().item())

        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
