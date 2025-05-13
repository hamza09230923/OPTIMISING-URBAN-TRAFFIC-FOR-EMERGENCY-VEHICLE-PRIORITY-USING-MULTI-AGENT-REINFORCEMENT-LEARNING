import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# ==== Q-Network for each agent ====
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        q = self.fc(out)
        return q, hidden

# ==== Mixing Network ====
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * 32)
        self.hyper_w2 = nn.Linear(state_dim, 32)
        self.hyper_b1 = nn.Linear(state_dim, 32)
        self.V = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        ts = agent_qs.size(1)
        states = states.view(bs * ts, -1)

        w1 = torch.abs(self.hyper_w1(states)).view(bs * ts, self.n_agents, 32)
        b1 = self.hyper_b1(states).view(bs * ts, 1, 32)

        agent_qs_reshaped = agent_qs.view(bs * ts, 1, self.n_agents)
        hidden = torch.bmm(agent_qs_reshaped, w1) + b1

        w2 = torch.abs(self.hyper_w2(states)).view(bs * ts, 32, 1)
        v = self.V(states).view(bs * ts, 1, 1)
        y = torch.bmm(hidden, w2) + v
        return y.view(bs, ts, 1)


# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return zip(*samples)

    def __len__(self):
        return len(self.buffer)

# ==== QMIX Trainer ====
class QMIXTrainer:
    def __init__(self, env, obs_dim=64, state_dim=64, n_actions=4, gamma=0.99, lr=1e-3):
        self.env = env
        self.n_agents = len(env.agents)
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma

        self.q_networks = [QNetwork(obs_dim, n_actions) for _ in range(self.n_agents)]
        self.target_q_networks = [QNetwork(obs_dim, n_actions) for _ in range(self.n_agents)]
        self.mixer = MixingNetwork(self.n_agents, state_dim)
        self.target_mixer = MixingNetwork(self.n_agents, state_dim)

        self.q_optimizers = [optim.Adam(q.parameters(), lr=lr) for q in self.q_networks]
        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=lr)

        self.buffer = ReplayBuffer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for net in self.q_networks + self.target_q_networks:
            net.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

    def select_actions(self, observations, epsilon=0.1):
        actions = {}
        for i, agent in enumerate(self.env.agents):
            obs = torch.FloatTensor(observations[agent]).unsqueeze(0).unsqueeze(0).to(self.device)
            q_vals, _ = self.q_networks[i](obs)
            if random.random() < epsilon:
                actions[agent] = random.randint(0, self.n_actions - 1)
            else:
                actions[agent] = q_vals.squeeze().argmax().item()
        return actions

    def train(self, episodes=1000, batch_size=32, update_target_every=10):
        for ep in range(episodes):
            obs = self.env.reset()
            done = False
            ep_transitions = []

            while not done:
                actions = self.select_actions(obs)
                next_obs, rewards, dones, _ = self.env.step(actions)

                state = np.concatenate([obs[a] for a in self.env.agents])
                next_state = np.concatenate([next_obs[a] for a in self.env.agents])
                reward = list(rewards.values())[0]
                done_flag = dones["__all__"]

                self.buffer.add((obs, actions, reward, next_obs, state, next_state, done_flag))
                obs = next_obs
                done = done_flag

            if len(self.buffer) >= batch_size:
                self.update(batch_size)

            if ep % update_target_every == 0:
                for i in range(self.n_agents):
                    self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())

            print(f"✅ Episode {ep+1}/{episodes} complete")

    def update(self, batch_size):
        batch = random.sample(self.buffer.buffer, batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, state_batch, next_state_batch, done_batch = zip(*batch)

        agent_qs, agent_next_qs = [], []

        for i in range(self.n_agents):
            obs_i = torch.FloatTensor([o[self.env.agents[i]] for o in obs_batch]).unsqueeze(1).to(self.device)
            next_obs_i = torch.FloatTensor([o[self.env.agents[i]] for o in next_obs_batch]).unsqueeze(1).to(self.device)

            q_vals, _ = self.q_networks[i](obs_i)
            next_q_vals, _ = self.target_q_networks[i](next_obs_i)

            actions = torch.LongTensor([a[self.env.agents[i]] for a in act_batch]).to(self.device)
            chosen_q = q_vals.gather(-1, actions.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)
            max_next_q = next_q_vals.max(dim=-1)[0]

            agent_qs.append(chosen_q)
            agent_next_qs.append(max_next_q)

        agent_qs = torch.stack(agent_qs, dim=2)
        agent_next_qs = torch.stack(agent_next_qs, dim=2)

        states = torch.FloatTensor(state_batch).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_state_batch).unsqueeze(1).to(self.device)

        mixed_q = self.mixer(agent_qs, states)
        target_mixed_q = self.target_mixer(agent_next_qs, next_states)

        rewards = torch.FloatTensor(rew_batch).unsqueeze(-1).unsqueeze(-1).to(self.device)
        dones = torch.FloatTensor(done_batch).unsqueeze(-1).unsqueeze(-1).to(self.device)
        targets = rewards + self.gamma * (1 - dones) * target_mixed_q

        loss = nn.MSELoss()(mixed_q, targets.detach())

        self.mixer_optimizer.zero_grad()
        for opt in self.q_optimizers:
            opt.zero_grad()
        loss.backward()
        self.mixer_optimizer.step()
        for opt in self.q_optimizers:
            opt.step()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for i, q in enumerate(self.q_networks):
            torch.save(q.state_dict(), f"{path}/agent_q{i}.pt")
        torch.save(self.mixer.state_dict(), f"{path}/mixer.pt")

    def load(self, path):
        for i, q in enumerate(self.q_networks):
            q.load_state_dict(torch.load(f"{path}/agent_q{i}.pt"))
        self.mixer.load_state_dict(torch.load(f"{path}/mixer.pt"))
