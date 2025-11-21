import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import os
from controller.Controller import Controller

# ---------------- Dueling DQN ------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + advantages - advantages.mean(1, keepdim=True)

# ---------------- Prioritized Replay Buffer ------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
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

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        states = torch.FloatTensor([np.concatenate([e.state[0].flatten(), [e.state[1]]]) for e in experiences])
        next_states = torch.FloatTensor([np.concatenate([e.next_state[0].flatten(), [e.next_state[1]]]) for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([e.reward for e in experiences])
        dones = torch.FloatTensor([float(e.done) for e in experiences])
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# ---------------- Advanced Rainbow DQN Controller ------------------
class AdvancedDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="rainbow_dqn.pth"):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("Advanced Rainbow DQN initialized...")
        self.state_dim = 5*5 + 1
        self.action_dim = len(self.directions)

        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        self.gamma = 0.99
        self.batch_size = 64
        self.buffer = PrioritizedReplayBuffer(30000)
        self.target_update_freq = 1000
        self.step_count = 0
        self.gradient_clip = 1.0

        # Multi-step buffer
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 1e-5

        if not self.is_training:
            self.load_model()
            self.q_network.eval()

    # ---------------- Epsilon-greedy decision ------------------
    def make_decision(self, robot, obstacles):
        state, distance_to_goal = robot.get_state(obstacles, 32, 32, self.goal)
        state = np.concatenate([state.flatten(), [distance_to_goal]])
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if self.is_training and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim-1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        return self.directions[action_idx]

    # ---------------- Store experience + n-step ------------------
    def store_experience(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        reward_sum, next_s, done_flag = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            reward_sum += (self.gamma ** idx) * r
            if d: 
                next_s, done_flag = _, True
                break
        s0, a0 = self.n_step_buffer[0][:2]
        self.buffer.add(s0, a0, reward_sum, next_s, done_flag)

    # ---------------- Train ------------------
    def train(self, beta=0.4):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(self.batch_size, beta)

        current_q = self.q_network(states).gather(1, actions).squeeze()
        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, best_actions).squeeze()
            target_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q

        td_errors = target_q - current_q
        loss = (td_errors.pow(2) * weights.to(self.device)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    # ---------------- Reward function (simple) ------------------
    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        position = (robot.grid_x, robot.grid_y)
        # Khởi tạo số lần đến vị trí này
        visit_count = getattr(self, 'visit_counts', {})
        visit_count[position] = visit_count.get(position, 0) + 1
        self.visit_counts = visit_count

        # Repeat penalty
        repeat_penalty = -0.05 * (visit_count[position] - 1)

        # Collision penalty
        collision_penalty = -0.2 if robot.check_collision(obstacles) else 0

        # Goal reward
        goal_reward = 1.0 if reached_goal else 0

        # Progress reward
        if prev_distance is None:
            progress_reward = 0
        else:
            delta = prev_distance - distance_to_goal
            progress_reward = delta * 5
            if delta < 0:  # đi xa hơn đích
                progress_reward -= 2

        # Tổng reward
        reward = progress_reward + collision_penalty + repeat_penalty + goal_reward
        reward = np.clip(reward, -1.0, 1.0)

        return reward

    # ---------------- Save / Load ------------------
    def _save_model_implementation(self):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.model_path)

    def _load_model_implementation(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            if self.is_training and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.q_network.eval()
