import numpy as np
import torch
import pygame
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
import os
from controller.Controller import Controller
from utils.gpu_utils import find_free_gpu

# ==============================================================================
# Prioritized Experience Replay (Giữ nguyên từ thuật toán 1)
# ==============================================================================
class SumTree:
    """Sum Tree for efficient sampling in PER"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.pending_idx + self.capacity - 1
        self.data[self.pending_idx] = data
        self.update(idx, p)
        self.pending_idx += 1
        if self.pending_idx >= self.capacity:
            self.pending_idx = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayMemory:
    """Prioritized Experience Replay Memory"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        self.max_priority = 1.0

    def add(self, experience):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if data is not None and isinstance(data, tuple):
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)
        
        if len(batch) == 0:
            return None, None, None

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, batch_idxs, batch_priorities):
        for idx, priority in zip(batch_idxs, batch_priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

# ==============================================================================
# Kiến trúc Mạng Nơ-ron MỚI: Transformer + Dueling DQN
# ==============================================================================
class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, sequence_length=10, d_model=128, nhead=4, num_encoder_layers=2, dim_feedforward=256):
        super(TransformerDuelingDQN, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # 1. Input Embedding: Project input_dim to the model's dimension (d_model)
        self.input_embed = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 4. Dueling Architecture Heads
        # Feature extraction after Transformer
        self.feature = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Embed input and add positional encoding
        x = self.input_embed(x) # -> (batch_size, seq_len, d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # PositionalEncoding expects (seq_len, batch, dim)
        
        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(x) # -> (batch_size, seq_len, d_model)
        
        # We only use the output of the last element in the sequence for the decision
        x = transformer_out[:, -1, :] # -> (batch_size, d_model)
        
        # Pass through Dueling heads
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# ==============================================================================
# Controller Chính
# ==============================================================================
class TransformerDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=True, model_path="transformer_dqn_model.pth"):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()
        if not self.is_training:
            self.load_model()

    def _get_valid_action_mask(self, robot, obstacles):
        mask = [True] * self.action_dim
        for i, direction in enumerate(self.directions):
            next_x = robot.x + direction[0] * self.cell_size
            next_y = robot.y + direction[1] * self.cell_size
            
            if not (self.env_padding <= next_x < self.env_padding + self.grid_width * self.cell_size and
                    self.env_padding <= next_y < self.env_padding + self.grid_height * self.cell_size):
                mask[i] = False
                continue
            
            for obs in obstacles:
                obstacle_rect = pygame.Rect(obs.x, obs.y, obs.width, obs.height)
                if obs.static and obstacle_rect.colliderect(pygame.Rect(next_x - robot.radius, next_y - robot.radius, robot.radius*2, robot.radius*2)):
                    mask[i] = False
                    break
        return mask

    def _initialize_algorithm(self):
        print("Initializing Controller with Transformer, PER, and Multi-Step DDQN...")
        self.state_dim = 5 * 5 + 1  # 5x5 matrix + distance to goal
        self.action_dim = len(self.directions)
        self.sequence_length = 10  # Dài hơn một chút để Transformer phát huy
        gpu_id = find_free_gpu()
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # Khởi tạo mạng Transformer
        self.q_network = TransformerDuelingDQN(self.state_dim, self.action_dim, self.sequence_length).to(self.device)
        self.target_network = TransformerDuelingDQN(self.state_dim, self.action_dim, self.sequence_length).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0 if self.is_training else 0.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9998 # Giảm epsilon chậm hơn một chút
        self.batch_size = 64
        self.target_update_freq = 250 # Cập nhật target network chậm hơn
        self.step_count = 0

        # Prioritized Experience Replay
        self.memory = PrioritizedReplayMemory(capacity=50000, beta_increment=0.0005)

        # Multi-step learning
        self.n_steps = 3
        self.multi_step_buffer = deque(maxlen=self.n_steps)
        self.gamma_n = self.gamma ** self.n_steps

        # Sequence management
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.padding_state = np.zeros(self.state_dim)

        # Anti-trap and curiosity (giữ nguyên)
        self.position_history = {}
        self.position_memory_size = 100
        self.position_history_list = []
        self.visit_counts = {}
        self.curiosity_factor = 0.1

    def make_decision(self, robot, obstacles):
        state, distance_to_goal = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal)
        state_flat = state.flatten()
        combined_state = np.concatenate([state_flat, [distance_to_goal]])

        self.sequence_buffer.append(combined_state)
        
        # Tạo chuỗi với padding nếu cần
        if len(self.sequence_buffer) < self.sequence_length:
            padding_count = self.sequence_length - len(self.sequence_buffer)
            padded_sequence = [self.padding_state] * padding_count + list(self.sequence_buffer)
        else:
            padded_sequence = list(self.sequence_buffer)

        sequence = np.array(padded_sequence)
        state_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Lấy action mask
        valid_mask = self._get_valid_action_mask(robot, obstacles)

        if self.is_training and random.random() < self.epsilon:
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
            action_idx = random.choice(valid_indices) if valid_indices else random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                self.q_network.eval()
                q_values = self.q_network(state_tensor)
                invalid_actions = torch.tensor([not v for v in valid_mask], dtype=torch.bool).to(self.device)
                q_values[0][invalid_actions] = -1e8
                action_idx = q_values.argmax().item()
                self.q_network.train()

        return self.directions[action_idx]

    def store_experience(self, state, action_idx, reward, next_state, done):
        state_matrix, state_distance = state
        next_state_matrix, next_state_distance = next_state
        state_flat = state_matrix.flatten()
        next_state_flat = next_state_matrix.flatten()
        state_combined = np.concatenate([state_flat, [state_distance]])
        next_state_combined = np.concatenate([next_state_flat, [next_state_distance]])

        self.multi_step_buffer.append((state_combined, action_idx, reward, next_state_combined, done))

        if len(self.multi_step_buffer) == self.n_steps or done:
            n_step_reward = 0
            gamma_power = 1.0
            for i in range(len(self.multi_step_buffer)):
                s, a, r, ns, d = self.multi_step_buffer[i]
                n_step_reward += gamma_power * r
                if d:
                    # Nếu tập kết thúc sớm, gamma_n được điều chỉnh
                    self.gamma_n = self.gamma ** (i + 1)
                    break
                gamma_power *= self.gamma
            else:
                 self.gamma_n = self.gamma ** self.n_steps

            # Tạo chuỗi trạng thái và trạng thái tiếp theo
            current_sequence = list(self.sequence_buffer)
            if len(current_sequence) < self.sequence_length:
                 state_sequence = [self.padding_state] * (self.sequence_length - len(current_sequence)) + current_sequence
            else:
                 state_sequence = current_sequence

            next_state_sequence = state_sequence[1:] + [next_state_combined]
            
            state_sequence = np.array(state_sequence)
            next_state_sequence = np.array(next_state_sequence)
            
            n_step_action = self.multi_step_buffer[0][1]
            n_step_done = self.multi_step_buffer[-1][4]

            experience = (state_sequence, n_step_action, n_step_reward, next_state_sequence, n_step_done)
            self.memory.add(experience)

            if done:
                self.multi_step_buffer.clear()
                self.sequence_buffer.clear() # Reset chuỗi cho tập mới

    def train(self):
        if len(self.memory) < self.batch_size:
            return
    
        batch, batch_idxs, importance_weights = self.memory.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        importance_weights = torch.FloatTensor(importance_weights).to(self.device)
    
        # Lệnh gọi mạng giờ không trả về hidden state
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
    
        with torch.no_grad():
            next_q_values_main = self.q_network(next_states)
            next_actions = next_q_values_main.argmax(dim=1, keepdim=True)
            
            next_q_values_target = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions).squeeze()
            
            target_q_values = rewards + (1 - dones) * self.gamma_n * next_q_values
    
        td_errors = torch.abs(current_q_values - target_q_values)
        loss = (importance_weights * (current_q_values - target_q_values).pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        priorities = td_errors.detach().cpu().numpy()
        self.memory.update_priorities(batch_idxs, priorities)
    
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    # Các hàm còn lại được giữ nguyên từ thuật toán 1
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        position = (robot.grid_x, robot.grid_y)
        
        if position in self.position_history:
            self.position_history[position] += 1
        else:
            self.position_history[position] = 1
        self.position_history_list.append(position)
        if len(self.position_history_list) > self.position_memory_size:
            old_pos = self.position_history_list.pop(0)
            if old_pos in self.position_history:
                self.position_history[old_pos] -= 1
                if self.position_history[old_pos] <= 0:
                    del self.position_history[old_pos]
        
        # SỬA: Tăng cường độ phạt lặp lại một cách mạnh mẽ
        repetition_penalty = -3.0 * (self.position_history.get(position, 1) - 1)
        
        # MỚI: Thêm hình phạt cho sự trì trệ (Stagnation Penalty)
        stagnation_penalty = 0
        history_length = 25 # Xem xét 25 bước đi gần nhất
        if len(self.position_history_list) > history_length:
            recent_history = self.position_history_list[-history_length:]
            num_unique_positions = len(set(recent_history))
            # Nếu robot chỉ loanh quanh ở 5 ô hoặc ít hơn, phạt rất nặng
            if num_unique_positions <= 5:
                stagnation_penalty = -5.0
        
        if position in self.visit_counts:
            self.visit_counts[position] += 1
        else:
            self.visit_counts[position] = 1
        curiosity_reward = self.curiosity_factor / max(1, self.visit_counts[position]**0.5)

        if reached_goal:
            return 100.0
        if robot.check_collision(obstacles):
            return -50.0
        
        if prev_distance is not None:
            progress_reward = (prev_distance - distance_to_goal) * 10
            step_penalty = -0.1
            if distance_to_goal > prev_distance:
                progress_reward -= 5
            return progress_reward + step_penalty + repetition_penalty + curiosity_reward + stagnation_penalty
        
        return -0.1 - (distance_to_goal * 0.05) + repetition_penalty + curiosity_reward + stagnation_penalty

    def save_model(self):
        torch.save(self.q_network.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.q_network.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.q_network.eval()
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file {self.model_path} not found!")