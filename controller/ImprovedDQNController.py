# controller/ImprovedDQNController.py

import numpy as np
import torch
import pygame
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import os
from controller.Controller import Controller
from utils.gpu_utils import find_free_gpu # MỚI: Tích hợp tìm GPU tự động

# --- Prioritized Replay Buffer (đơn giản) ---
# ... (Giữ nguyên toàn bộ class PrioritizedReplayBuffer và DuelingDQN)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(self.Transition(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = self.Transition(state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None, None, None

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        batch = self.Transition(*zip(*samples))
        return batch, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)

# --- Network architecture: Dueling DQN ---
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


class ImprovedDQNController(Controller):
    # SỬA: Thêm grid_width và grid_height vào constructor
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=True, model_path="improved_dqn_model.pth"):
        # MỚI: Lưu lại grid size
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()
    
    def _initialize_algorithm(self):
        self.state_dim = 5 * 5 + 1
        self.action_dim = len(self.directions)
        
        # MỚI: Tự động chọn GPU rảnh nhất
        gpu_id = find_free_gpu()
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")
        
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)
        
        self.gamma = 0.99
        self.epsilon = 1.0 if self.is_training else 0.0
        self.epsilon_min = 0.01 # SỬA: Giảm epsilon_min để tăng khả năng khai thác khi đã học tốt
        self.epsilon_decay = 0.999 # SỬA: Giảm chậm hơn một chút
        self.batch_size = 64
        
        self.memory = PrioritizedReplayBuffer(capacity=20000, alpha=0.6) # SỬA: Tăng capacity
        self.beta_start = 0.4
        self.beta_frames = 100000 # Số frame để beta tăng từ start -> 1.0
        
        self.target_update_tau = 1e-3
        self.frame_idx = 0
        
        self.position_history = {}
        self.position_memory_size = 100
        self.position_history_list = []
        self.visit_counts = {}
        self.curiosity_factor = 0.05
        
        if not self.is_training:
            self.load_model()
    
    def make_decision(self, robot, obstacles):
        # SỬA: Sử dụng self.grid_width và self.grid_height đã được lưu
        state, distance_to_goal = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal)
        state_flat = state.flatten()
        combined_state = np.concatenate([state_flat, [distance_to_goal]])
        state_tensor = torch.FloatTensor(combined_state).unsqueeze(0).to(self.device)
        
        # MỚI: Sử dụng action masking để tránh các hành động không hợp lệ
        valid_mask = self._get_valid_action_mask(robot, obstacles)
        
        if self.is_training and random.random() < self.epsilon:
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
            action_idx = random.choice(valid_indices) if valid_indices else random.randint(0, self.action_dim-1)
        else:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                # Áp dụng mask
                invalid_actions = torch.tensor([not v for v in valid_mask], dtype=torch.bool).to(self.device)
                q_values[0][invalid_actions] = -1e8 # Gán giá trị rất nhỏ cho hành động không hợp lệ
                action_idx = q_values.argmax(dim=1).item()
        
        self.q_network.train() # MỚI: Luôn chuyển về mode train sau khi ra quyết định
        return self.directions[action_idx]
    
    def _get_valid_action_mask(self, robot, obstacles):
        # SỬA: Implement logic đơn giản cho action masking
        mask = [True] * self.action_dim
        for i, direction in enumerate(self.directions):
            # Giả lập vị trí tiếp theo
            next_x = robot.x + direction[0] * self.cell_size
            next_y = robot.y + direction[1] * self.cell_size
            
            # Kiểm tra va chạm với tường
            if not (self.env_padding <= next_x < self.env_padding + self.grid_width * self.cell_size and
                    self.env_padding <= next_y < self.env_padding + self.grid_height * self.cell_size):
                mask[i] = False
                continue
            
            # Kiểm tra va chạm với vật cản tĩnh (đơn giản hóa)
            for obs in obstacles:
                obstacle_rect = pygame.Rect(obs.x, obs.y, obs.width, obs.height)
                if obs.static and obstacle_rect.colliderect(pygame.Rect(next_x - robot.radius, next_y - robot.radius, robot.radius*2, robot.radius*2)):
                    mask[i] = False
                    break
        return mask
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        state_matrix, state_distance = state
        next_matrix, next_distance = next_state
        state_flat = state_matrix.flatten()
        next_flat = next_matrix.flatten()
        state_combined = np.concatenate([state_flat, [state_distance]])
        next_combined = np.concatenate([next_flat, [next_distance]])
        self.memory.push(state_combined, action_idx, reward, next_combined, done)
    
    # SỬA: Thay đổi chữ ký hàm train để tương thích với main.py
    def train(self): # XÓA: Tham số frame_idx
        self.frame_idx += 1
        if len(self.memory) < self.batch_size:
            return
        
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        batch, indices, weights = self.memory.sample(self.batch_size, beta)

        if batch is None: # MỚI: Thêm kiểm tra nếu sample không thành công
            return
        
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values
        
        # SỬA: Dùng Huber loss (SmoothL1Loss) để ổn định hơn
        loss = nn.SmoothL1Loss(reduction='none')(q_values, target_q_values)
        loss = (loss * weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0) # SỬA: Giảm max_norm
        self.optimizer.step()
        
        td_errors = (target_q_values - q_values).abs().detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.target_update_tau * param.data + (1.0 - self.target_update_tau) * target_param.data)
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        # ... (Giữ nguyên logic tính reward)
        position = (robot.grid_x, robot.grid_y)
        if position in self.position_history:
            self.position_history[position] += 1
        else:
            self.position_history[position] = 1
        
        self.position_history_list.append(position)
        if len(self.position_history_list) > self.position_memory_size:
            old = self.position_history_list.pop(0)
            if old in self.position_history:
                self.position_history[old] -= 1
                if self.position_history[old] <= 0:
                    del self.position_history[old]
        
        repetition_penalty = min(-0.5 * (self.position_history.get(position, 1) - 1), 0)
        
        if position in self.visit_counts:
            self.visit_counts[position] += 1
        else:
            self.visit_counts[position] = 1
        curiosity_reward = self.curiosity_factor / (self.visit_counts[position]**0.5)
        
        if reached_goal:
            return 100.0
        if robot.check_collision(obstacles):
            return -50.0
        
        if prev_distance is not None:
            progress_reward = (prev_distance - distance_to_goal) * 10.0
            step_penalty = -0.1
            if distance_to_goal > prev_distance:
                progress_reward -= 2.0
            return progress_reward + step_penalty + repetition_penalty + curiosity_reward
        
        return -0.1 - (distance_to_goal * 0.05) + repetition_penalty + curiosity_reward

    # SỬA: Đổi tên hàm để tương thích với base class Controller
    def save_model(self):
        torch.save(self.q_network.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        if os.path.exists(self.model_path):
            self.q_network.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.q_network.eval()
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")