import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from controller.Controller import Controller


class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration (replaces epsilon-greedy)"""

    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class DuelingNoisyDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNoisyDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(64, 32),
            nn.ReLU(),
            NoisyLinear(32, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(64, 32),
            nn.ReLU(),
            NoisyLinear(32, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class AdvancedDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="advanced_dqn_model.pth"):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("Advanced DQN initialized...")
        self.state_dim = 5 * 5 + 1
        self.action_dim = len(self.directions)

        # Networks with Noisy layers
        self.q_network = DuelingNoisyDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingNoisyDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer + Scheduler
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005, eps=1e-8)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        # Hyperparameters
        self.gamma = 0.99
        self.batch_size = 64
        self.memory = deque(maxlen=20000)
        self.memory.clear()
        self.target_update_freq = 200
        self.step_count = 0
        self.gradient_clip = 1.0

        # Anti-trap & Curiosity
        self.position_history = {}
        self.position_history_list = []
        self.position_memory_size = 100
        self.visit_counts = {}
        self.curiosity_factor = 0.15  # Tăng curiosity

        # Load model if testing
        if not self.is_training:
            self.load_model()
            self.q_network.eval()

    def make_decision(self, robot, obstacles):
        state, distance_to_goal = robot.get_state(obstacles, 32, 32, self.goal)
        state_flat = state.flatten()
        combined_state = np.concatenate([state_flat, [distance_to_goal]])
        state_tensor = torch.FloatTensor(combined_state).unsqueeze(0).to(self.device)

        self.q_network.train() if self.is_training else self.q_network.eval()
        self.q_network.reset_noise()  # Reset noise for exploration

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()

        return self.directions[action_idx]

    def store_experience(self, state, action_idx, reward, next_state, done):
        state_matrix, state_distance = state
        next_state_matrix, next_state_distance = next_state
        state_flat = state_matrix.flatten()
        next_state_flat = next_state_matrix.flatten()
        state_combined = np.concatenate([state_flat, [state_distance]])
        next_state_combined = np.concatenate([next_state_flat, [next_state_distance]])

        state_tensor = torch.FloatTensor(state_combined).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state_combined).to(self.device)
        action = torch.LongTensor([action_idx]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([float(done)]).to(self.device)

        self.memory.append((state_tensor, action, reward, next_state_tensor, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards).squeeze()
        next_states = torch.stack(next_states)
        dones = torch.stack(dones).squeeze()

        # Double DQN
        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()

        # Reset noise after update
        self.q_network.reset_noise()

        # Update target
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        pass  # No epsilon → Noisy Networks handles exploration

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        position = (robot.grid_x, robot.grid_y)

        # Update history
        self.position_history[position] = self.position_history.get(position, 0) + 1
        self.position_history_list.append(position)
        if len(self.position_history_list) > self.position_memory_size:
            old = self.position_history_list.pop(0)
            self.position_history[old] -= 1
            if self.position_history[old] <= 0:
                del self.position_history[old]

        # Visit count
        self.visit_counts[position] = self.visit_counts.get(position, 0) + 1
        visit = self.visit_counts[position]

        # Curiosity bonus
        curiosity = self.curiosity_factor / (visit ** 0.6 + 1e-3)

        # Repetition penalty (stronger for repeated visits)
        repeat_penalty = -0.5 * (self.position_history.get(position, 1) - 1) ** 1.5

        if reached_goal:
            return 100 + curiosity * 10

        if robot.check_collision(obstacles):
            return -50 + repeat_penalty

        if prev_distance is None:
            return -0.2 - 0.03 * distance_to_goal + curiosity + repeat_penalty

        # Progress reward
        delta = prev_distance - distance_to_goal
        progress = delta * 12
        if delta < 0:
            progress -= 8  # Strong penalty for moving away

        step_penalty = -0.15
        total = progress + step_penalty + curiosity + repeat_penalty

        return np.clip(total, -20, 50)  # Clip extreme values

    def _save_model_implementation(self):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, self.model_path)

    def _load_model_implementation(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            if self.is_training and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.q_network.eval()