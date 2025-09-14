# td3_caching.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from collections import deque, namedtuple

# -------------------------
# Actor / Critic Networks
# -------------------------
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        a = torch.sigmoid(self.l3(x))  # we want [0,1] priorities
        return a * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        self.apply(weights_init_)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], dim=1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        return self.l3(x1)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.storage = []

    def add(self, data):
        if self.size < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        s, a, r, s2, d = [], [], [], [], []
        for i in ind:
            si, ai, ri, s2i, di = self.storage[i]
            s.append(np.array(si, copy=False))
            a.append(np.array(ai, copy=False))
            r.append(np.array(ri, copy=False))
            s2.append(np.array(s2i, copy=False))
            d.append(np.array(di, copy=False))
        return (
            torch.FloatTensor(np.array(s)),
            torch.FloatTensor(np.array(a)),
            torch.FloatTensor(np.array(r)).unsqueeze(1),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(np.array(d)).unsqueeze(1),
        )

# -------------------------
# TD3 Agent
# -------------------------
class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        device='cpu',
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=1e-3,
        critic_lr=1e-3
    ):
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, 0.0, self.max_action)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(0.0, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# -------------------------
# Environment Stub
# -------------------------
class EdgeCachingEnv:
    """
    A simple environment that simulates caching decisions for `num_models`.
    Replace or connect this to your iFogSim run loop: at each env.step(action),
    you should query simulated user requests within that time-step and compute
    latency/network cost/model download cost to compute reward.
    """

    def __init__(self, num_models=4, history_len=10, cache_capacity=2, seed=0):
        np.random.seed(seed)
        self.num_models = num_models
        self.history_len = history_len
        self.cache_capacity = cache_capacity

        # internal state
        # request_history: last `history_len` counts for each model
        self.request_history = np.zeros((self.num_models, self.history_len), dtype=np.float32)
        # cache vector: 1 if model is cached at edge server
        self.cache = np.zeros(self.num_models, dtype=np.int32)
        # network bandwidth / latency proxies
        self.bandwidth = 10.0  # Mbps (example)
        self.latency_base = 50.0  # ms to cloud (example)
        self.t = 0

    def reset(self):
        self.request_history.fill(0.0)
        self.cache.fill(0)
        self.bandwidth = 10.0
        self.latency_base = 50.0
        self.t = 0
        return self._get_state()

    def _get_state(self):
        # State vector ideas:
        # 1) flattened request history (num_models * history_len)
        # 2) cache binary vector (num_models)
        # 3) normalized bandwidth & latency scalars (2)
        s = np.concatenate([
            self.request_history.flatten() / (1.0 + np.max(self.request_history)),
            self.cache.astype(np.float32),
            np.array([self.bandwidth / 100.0, self.latency_base / 500.0], dtype=np.float32)
        ])
        return s

    def sample_user_requests(self):
        """
        Simulate incoming request counts for each model in this time-step.
        Replace this with actual counts from iFogSim events.
        """
        # Example: Poisson-like spikes for different models
        lam = np.array([0.8, 0.6, 0.4, 0.2]) * (1 + 0.5 * np.sin(self.t / 10.0))
        reqs = np.random.poisson(lam=lam).astype(np.float32)
        return reqs

    def step(self, action):
        """
        action: continuous vector in [0,1] per model representing priority.
        Environment will pick top-k models to be cached given capacity.
        Returns: next_state, reward, done, info
        """
        self.t += 1
        action = np.clip(action, 0.0, 1.0)

        # Convert continuous action to discrete caching decision: top-K
        k = self.cache_capacity
        topk_idx = np.argsort(-action)[:k]
        new_cache = np.zeros_like(self.cache)
        new_cache[topk_idx] = 1

        # Compute model load/unload cost (if uncached -> cache => download cost)
        downloads = np.logical_and(new_cache == 1, self.cache == 0).sum()
        download_cost = downloads * 10.0  # arbitrary cost unit for network usage

        # Update cache
        self.cache = new_cache

        # Get incoming requests
        reqs = self.sample_user_requests()
        # update history
        self.request_history = np.roll(self.request_history, -1, axis=1)
        self.request_history[:, -1] = reqs

        # For each model, if cached -> edge latency small, else cloud latency
        edge_latency_per_req = 20.0  # ms
        cloud_latency_per_req = self.latency_base + 50.0  # ms

        total_latency = 0.0
        hits = 0
        total_reqs = reqs.sum() + 1e-6
        for i in range(self.num_models):
            if self.cache[i] == 1:
                total_latency += reqs[i] * edge_latency_per_req
                hits += reqs[i]
            else:
                total_latency += reqs[i] * cloud_latency_per_req

        avg_latency = total_latency / total_reqs

        # Reward shaping:
        # We want higher cache hit (positive), lower latency (negative), lower download cost (negative)
        hit_rate = hits / total_reqs
        reward = (hit_rate * 2.0) - (avg_latency / 1000.0) - (download_cost / 100.0)

        # small penalty to avoid churning caches every step
        # heavy changes between previous and current cache
        churn = np.sum(np.abs(self.cache - new_cache))
        reward -= churn * 0.01

        next_state = self._get_state()
        done = False  # episodic logic can be added externally
        info = {
            'avg_latency_ms': avg_latency,
            'hit_rate': hit_rate,
            'download_cost': download_cost,
            'downloads': int(downloads)
        }
        return next_state, reward, done, info

# -------------------------
# Training Loop Skeleton
# -------------------------
if __name__ == "__main__":
    # Hyperparams / Setup
    num_models = 4
    history_len = 10
    cache_capacity = 2
    state_dim = num_models * history_len + num_models + 2
    action_dim = num_models
    max_action = 1.0

    env = EdgeCachingEnv(num_models=num_models, history_len=history_len, cache_capacity=cache_capacity)
    replay = ReplayBuffer(max_size=int(1e5))
    agent = TD3Agent(state_dim, action_dim, max_action, device='cpu')

    episodes = 400
    episode_length = 50
    start_timesteps = 1000
    batch_size = 256
    expl_noise = 0.1  # exploration noise for actions
    total_steps = 0

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        for t in range(episode_length):
            total_steps += 1
            if total_steps < start_timesteps:
                action = np.random.rand(action_dim)  # random priorities
            else:
                action = agent.select_action(state, noise=expl_noise)

            next_state, reward, done, info = env.step(action)
            replay.add((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward

            if replay.size > batch_size and total_steps >= start_timesteps:
                agent.train(replay, batch_size=batch_size)

            if done:
                break

        if ep % 10 == 0:
            print(f"Episode {ep:03d} | Reward: {ep_reward:.3f} | Avg latency(ms): {info['avg_latency_ms']:.2f} | Hit rate: {info['hit_rate']:.3f}")

    print("Training loop finished. Save models if desired.")
    # Example: torch.save(agent.actor.state_dict(), "actor.pth")
