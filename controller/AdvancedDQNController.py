import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import os
from controller.Controller import Controller

# ---------------- Noisy Linear (Fortunato) ------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)

        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        if bias:
            self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1. / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)

        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        epsilon_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            if self.bias_mu is not None:
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.bias_mu is not None else None
        return nn.functional.linear(input, weight, bias)


# ---------------- Dueling DQN (with optional Noisy) ------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, use_noisy=False):
        super().__init__()
        Linear = NoisyLinear if use_noisy else nn.Linear
        self.use_noisy = use_noisy

        self.feature = nn.Sequential(
            Linear(input_dim, 256),
            nn.ReLU(),
            Linear(256, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            Linear(256, 128),
            nn.ReLU(),
            Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            Linear(256, 128),
            nn.ReLU(),
            Linear(128, output_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(1, keepdim=True)

    def reset_noise(self):
        if not self.use_noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ---------------- Prioritized Replay Buffer ------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = int(capacity)
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.alpha = alpha
        self.experience = namedtuple('Experience',
                                     field_names=['state','action','reward','next_state','done'])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta, device):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs /= probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

        states_np = np.array(
            [np.concatenate([e.state[0].flatten(), [e.state[1]]]) for e in experiences],
            dtype=np.float32
        )
        next_states_np = np.array(
            [np.concatenate([e.next_state[0].flatten(), [e.next_state[1]]]) for e in experiences],
            dtype=np.float32
        )

        states = torch.tensor(states_np, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=device)
        dones = torch.tensor([float(e.done) for e in experiences], dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, prios):
        for i, p in zip(indices, prios):
            self.priorities[int(i)] = p

    def __len__(self):
        return len(self.buffer)


# ---------------- Advanced Rainbow DQN Controller ------------------
class AdvancedDQNController(Controller):
    def __init__(
        self,
        goal,
        cell_size,
        env_padding,
        is_training=True,
        model_path="rainbow_dqn.pth",
        use_noisy=False,
        soft_tau=0.005,          # if >0 use soft update every step; otherwise use hard update freq
        target_update_freq=1000
    ):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self.use_noisy = use_noisy
        self.soft_tau = soft_tau
        self.target_update_freq = target_update_freq
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("Advanced Rainbow DQN initialized... (Noisy=%s, tau=%.5f)" % (self.use_noisy, self.soft_tau))
        self.state_dim = 5*5 + 1
        self.action_dim = len(self.directions)

        # Networks
        self.q_network = DuelingDQN(self.state_dim, self.action_dim, use_noisy=self.use_noisy).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim, use_noisy=self.use_noisy).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)
        self.gamma = 0.99
        self.batch_size = 64
        self.buffer = PrioritizedReplayBuffer(30000)
        self.step_count = 0
        self.gradient_clip = 1.0

        # Multi-step
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 1e-5

        # PER / beta annealing
        self.beta_start = 0.4
        self.beta_frames = 1_000_000
        self.frame_idx = 0

        if not self.is_training:
            self._load_model_implementation()
            self.q_network.eval()

    # ---------------- Epsilon-greedy + Noisy handling ------------------
    def make_decision(self, robot, obstacles):
        s, dist = robot.get_state(obstacles, 32, 32, self.goal)
        s = np.concatenate([s.flatten(), [dist]])
        state = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

        # If using NoisyNet, we rely on parameter noise instead of epsilon-greedy.
        if self.use_noisy and self.is_training:
            # reset noise for exploration
            self.q_network.reset_noise()
            with torch.no_grad():
                return self.directions[self.q_network(state).argmax().item()]

        # Otherwise use epsilon-greedy
        if self.is_training and random.random() < self.epsilon:
            return self.directions[random.randint(0, self.action_dim-1)]
        else:
            with torch.no_grad():
                return self.directions[self.q_network(state).argmax().item()]

    # ---------------- Store experience (N-step) ------------------
    def store_experience(self, state, action, reward, next_state, done):
        """
        state: tuple or structure as before (state_grid, distance)
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return

        R, next_s, done_flag = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for t, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** t) * r
            if d:
                next_s = _
                done_flag = True
                break

        s0, a0 = self.n_step_buffer[0][:2]
        self.buffer.add(s0, a0, R, next_s, done_flag)

    # ---------------- Train ------------------
    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        # Beta annealing for PER
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame_idx / max(1, self.beta_frames)))
        self.frame_idx += 1

        states, actions, rewards, next_states, dones, weights, indices = \
            self.buffer.sample(self.batch_size, beta, self.device)

        # Reset noise in noisy layers if used
        if self.use_noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Current Q
        current_q = self.q_network(states).gather(1, actions).squeeze()

        with torch.no_grad():
            # Double DQN target
            best_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, best_actions).squeeze()
            target_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q

        td_errors = target_q - current_q
        loss = (td_errors.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()

        # Update PER priorities (use abs td + small eps)
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        # Epsilon decay (only if not using noisy)
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # Target update: soft or hard
        self.step_count += 1
        if self.soft_tau and self.soft_tau > 0:
            # soft update every step
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.soft_tau * param.data + (1.0 - self.soft_tau) * target_param.data)
        else:
            if self.step_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

    # ---------------- Reward function (improved) ------------------
    def calculate_reward(
            self,
            robot,
            obstacles,
            done,
            reached_goal,
            distance_to_goal,
            *,
            prev_distance=None
    ):
        pos = (robot.grid_x, robot.grid_y)

        # Track visit counts
        visits = getattr(self, 'visit_counts', {})
        visits[pos] = visits.get(pos, 0) + 1
        self.visit_counts = visits

        # Repeat penalty
        repeat_penalty = -0.05 * (visits[pos] - 1)

        # Collision penalty
        collision_penalty = -0.4 if robot.check_collision(obstacles) else 0

        # Goal reward
        goal_reward = 1.0 if reached_goal else 0.0

        # Progress reward
        if prev_distance is None:
            progress = 0.0
        else:
            delta = prev_distance - distance_to_goal
            progress = delta * 6.0

            if delta < 0:
                # punish moving away
                progress -= 2.0

        # Exploration bonus
        explore_bonus = 0.05 if visits[pos] == 1 else 0.0

        reward = progress + repeat_penalty + collision_penalty + goal_reward + explore_bonus
        return float(np.clip(reward, -1.0, 1.0))

    # ---------------- Save / Load ------------------
    def _save_model_implementation(self):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.model_path)

    def _load_model_implementation(self):
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=self.device)
            self.q_network.load_state_dict(ckpt['q_network'])
            self.target_network.load_state_dict(ckpt['q_network'])
            if self.is_training and 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            self.q_network.eval()