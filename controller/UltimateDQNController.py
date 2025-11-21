import numpy as np
import torch
import pygame
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque, namedtuple
import os
from controller.Controller import Controller
from utils.gpu_utils import find_free_gpu

# --- CÁC THÀNH PHẦN GIỮ NGUYÊN ---
# ... (Các class PrioritizedReplayBuffer và DuelingDQN giữ nguyên như trước) ...
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        # SỬA: Thêm action_onehot để lưu cho ICM
        self.Transition = namedtuple('Transition', ('state', 'action', 'action_onehot', 'reward', 'next_state', 'done', 'gamma_n'))
    
    def push(self, state, action, action_onehot, reward, next_state, done, gamma_n):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(self.Transition(state, action, action_onehot, reward, next_state, done, gamma_n))
        else:
            self.buffer[self.pos] = self.Transition(state, action, action_onehot, reward, next_state, done, gamma_n)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0: return None, None, None
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = self.Transition(*zip(*samples))
        return batch, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities): self.priorities[idx] = prio
    def __len__(self): return len(self.buffer)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))
    
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ==============================================================================
# MỚI: Intrinsic Curiosity Module (ICM)
# ==============================================================================
class ICMNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        
        # 1. Feature Encoder: Mã hóa trạng thái
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Inverse Model: Dự đoán hành động từ (trạng thái, trạng thái tiếp theo)
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 3. Forward Model: Dự đoán trạng thái tiếp theo từ (trạng thái, hành động)
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) # Dự đoán đặc trưng của trạng thái tiếp theo
        )
    
    def forward(self, state, action_onehot, next_state):
        # Mã hóa trạng thái
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Tính intrinsic reward từ Forward Model
        pred_next_state_feat = self.forward_model(torch.cat([state_feat, action_onehot], dim=1))
        forward_loss = 0.5 * ((pred_next_state_feat - next_state_feat.detach()) ** 2).sum(dim=1)
        
        # Tính loss cho Inverse Model
        pred_action = self.inverse_model(torch.cat([state_feat, next_state_feat], dim=1))
        
        return forward_loss, pred_action

# ==============================================================================
# CONTROLLER TỐI THƯỢNG (PHIÊN BẢN ICM)
# ==============================================================================
class UltimateDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=True, model_path="ultimate_icm_dqn_model.pth"):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()
    
    def _initialize_algorithm(self):
        print("Initializing Ultimate DQN Controller with ICM and Enhanced Vision...")
        # SỬA: Cập nhật state_dim cho tầm nhìn mở rộng
        self.state_dim = (5 * 5) + (3 * 3) + 1  # 5x5 chi tiết + 3x3 downsample + 1 khoảng cách
        self.action_dim = len(self.directions)
        
        gpu_id = find_free_gpu()
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")
        
        # Khởi tạo mạng chính
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4) # Giảm learning rate
        
        # MỚI: Khởi tạo mạng ICM và optimizer riêng
        self.icm_network = ICMNetwork(self.state_dim, self.action_dim).to(self.device)
        self.icm_optimizer = optim.Adam(self.icm_network.parameters(), lr=1e-4)
        self.intrinsic_reward_scale = 0.01 # Hệ số cho phần thưởng tò mò
        
        self.gamma = 0.99
        self.epsilon = 1.0; self.epsilon_min = 0.05; self.epsilon_decay = 0.9999 # Giảm epsilon rất chậm
        self.batch_size = 128 # Tăng batch size
        self.target_update_freq = 200
        
        self.memory = PrioritizedReplayBuffer(capacity=50000, alpha=0.5)
        self.beta_start = 0.4; self.beta_frames = 100000
        
        self.n_steps = 3
        self.multi_step_buffer = deque(maxlen=self.n_steps)
        self.step_count = 0
        self.position_history = {}; self.position_memory_size = 100; self.position_history_list = []
        
        if not self.is_training: self.load_model()
    
    # MỚI: Hàm lấy trạng thái mở rộng
    def _get_enhanced_state(self, robot, obstacles):
        # Vùng gần: 5x5 chi tiết
        close_state_matrix, distance_to_goal = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal, vision_size=5)
        
        # Vùng xa: 9x9
        far_state_matrix, _ = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal, vision_size=9)
        
        # Downsample vùng xa xuống 3x3 bằng cách lấy giá trị trung bình của các khối 3x3
        h, w = far_state_matrix.shape
        new_h, new_w = 3, 3
        shape = (new_h, h // new_h, new_w, w // new_w)
        far_state_downsampled = far_state_matrix.reshape(shape).mean(-1).mean(1)
        
        # Kết hợp thành vector trạng thái cuối cùng
        combined_state = np.concatenate([close_state_matrix.flatten(), far_state_downsampled.flatten(), [distance_to_goal]])
        return combined_state

    def make_decision(self, robot, obstacles):
        self.robot = robot
        self.obstacles = obstacles
        combined_state = self._get_enhanced_state(robot, obstacles)
        state_tensor = torch.FloatTensor(combined_state).unsqueeze(0).to(self.device)
        
        valid_mask = self._get_valid_action_mask(robot, obstacles)
        
        if self.is_training and random.random() < self.epsilon:
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
            action_idx = random.choice(valid_indices) if valid_indices else random.randint(0, self.action_dim-1)
        else:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                invalid_actions = torch.tensor([not v for v in valid_mask], dtype=torch.bool).to(self.device)
                q_values[0][invalid_actions] = -1e8
                action_idx = q_values.argmax(dim=1).item()
        
        self.q_network.train()
        return self.directions[action_idx]
    
    # ... (Hàm _get_valid_action_mask giữ nguyên) ...
    def _get_valid_action_mask(self, robot, obstacles):
        mask = [True] * self.action_dim
        for i, direction in enumerate(self.directions):
            next_x = robot.x + direction[0] * self.cell_size
            next_y = robot.y + direction[1] * self.cell_size
            if not (self.env_padding <= next_x < self.env_padding + self.grid_width * self.cell_size and
                    self.env_padding <= next_y < self.env_padding + self.grid_height * self.cell_size):
                mask[i] = False; continue
            for obs in obstacles:
                obstacle_rect = pygame.Rect(obs.x, obs.y, obs.width, obs.height)
                if obs.static and obstacle_rect.colliderect(pygame.Rect(next_x - robot.radius, next_y - robot.radius, robot.radius*2, robot.radius*2)):
                    mask[i] = False; break
        return mask

    def store_experience(self, state, action_idx, reward, next_state, done):
        # state và next_state nhận vào là các tuple (matrix_5x5, distance) từ main.py
        
        # --- BẮT ĐẦU KHỐI CODE SỬA ---
        # Tái tạo lại state_combined từ thông tin thô của `state`
        close_state_matrix, distance_to_goal = state
        # Để có far_state, chúng ta phải giả định rằng robot chưa di chuyển kể từ lần gọi get_state()
        # Đây là một giả định hợp lý trong luồng update() của main.py
        far_state_matrix, _ = self.robot.get_state(self.obstacles, self.grid_width, self.grid_height, self.goal, vision_size=9)
        h, w = far_state_matrix.shape
        shape = (3, h // 3, 3, w // 3)
        far_state_downsampled = far_state_matrix.reshape(shape).mean(-1).mean(1)
        state_combined = np.concatenate([close_state_matrix.flatten(), far_state_downsampled.flatten(), [distance_to_goal]])

        # Tái tạo lại next_combined từ thông tin thô của `next_state`
        # `next_state` được tính sau khi robot đã di chuyển, nên ta không cần gọi lại get_state
        next_close_state_matrix, next_distance_to_goal = next_state
        # Chúng ta cần tính lại far_state cho vị trí MỚI của robot
        next_far_state_matrix, _ = self.robot.get_state(self.obstacles, self.grid_width, self.grid_height, self.goal, vision_size=9)
        h, w = next_far_state_matrix.shape
        shape = (3, h // 3, 3, w // 3)
        next_far_state_downsampled = next_far_state_matrix.reshape(shape).mean(-1).mean(1)
        next_combined = np.concatenate([next_close_state_matrix.flatten(), next_far_state_downsampled.flatten(), [next_distance_to_goal]])
        # --- KẾT THÚC KHỐI CODE SỬA ---

        # MỚI: Tính intrinsic reward từ ICM
        intrinsic_reward = 0
        if self.is_training:
            state_tensor = torch.FloatTensor(state_combined).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_combined).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action_idx]).to(self.device)
            action_onehot = torch.zeros(1, self.action_dim).to(self.device)
            action_onehot.scatter_(1, action_tensor.unsqueeze(1), 1)

            with torch.no_grad():
                # Chỉ lấy forward_loss (lỗi dự đoán) làm phần thưởng
                forward_loss, _ = self.icm_network(state_tensor, action_onehot, next_state_tensor)
                intrinsic_reward = forward_loss.item() * self.intrinsic_reward_scale
        
        # Kết hợp phần thưởng
        combined_reward = reward + intrinsic_reward
        action_onehot_np = np.zeros(self.action_dim); action_onehot_np[action_idx] = 1

        self.multi_step_buffer.append((state_combined, action_idx, action_onehot_np, combined_reward))

        # Logic N-step learning được sửa đổi để xử lý action_onehot
        if len(self.multi_step_buffer) < self.n_steps and not done:
            return
        R = 0; gamma_n = 1.0
        for i in range(len(self.multi_step_buffer)):
            s, a, a_oh, r = self.multi_step_buffer[i]
            R += gamma_n * r
            gamma_n *= self.gamma
        
        first_state, first_action, first_action_onehot, _ = self.multi_step_buffer[0]
        final_gamma_n = self.gamma ** len(self.multi_step_buffer)
        
        self.memory.push(first_state, first_action, first_action_onehot, R, next_combined, done, final_gamma_n)
        
        if done:
            # Xử lý các kinh nghiệm còn lại trong buffer khi episode kết thúc
            while len(self.multi_step_buffer) > 1:
                self.multi_step_buffer.popleft()
                R = 0; gamma_n = 1.0
                for i in range(len(self.multi_step_buffer)):
                    s, a, a_oh, r = self.multi_step_buffer[i]
                    R += gamma_n * r
                    gamma_n *= self.gamma
                
                first_state, first_action, first_action_onehot, _ = self.multi_step_buffer[0]
                final_gamma_n = self.gamma ** len(self.multi_step_buffer)
                self.memory.push(first_state, first_action, first_action_onehot, R, next_combined, done, final_gamma_n)
            self.multi_step_buffer.clear()
    
    def train(self):
        self.step_count += 1
        if len(self.memory) < self.batch_size: return
        
        beta = min(1.0, self.step_count * (1.0 - self.beta_start) / self.beta_frames)
        batch, indices, weights = self.memory.sample(self.batch_size, beta)
        if batch is None: return
        
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        actions_onehot = torch.FloatTensor(np.array(batch.action_onehot)).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        gamma_ns = torch.FloatTensor(batch.gamma_n).unsqueeze(1).to(self.device)

        # 1. Huấn luyện mạng DQN (như cũ)
        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1.0 - dones) * gamma_ns * next_q_values
        
        dqn_loss = (weights * nn.SmoothL1Loss(reduction='none')(q_values, target_q_values)).mean()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 2. MỚI: Huấn luyện mạng ICM
        forward_loss, pred_action = self.icm_network(states, actions_onehot, next_states)
        inverse_loss = nn.CrossEntropyLoss()(pred_action, actions.squeeze())
        icm_loss = (0.8 * inverse_loss + 0.2 * forward_loss.mean())
        
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        # Cập nhật priorities và target network
        td_errors = (target_q_values - q_values).abs().detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        # --- HÀM PHẦN THƯỞNG ĐƯỢC NÂNG CẤP TOÀN DIỆN ---
        
        # 1. Hình phạt va chạm và Phần thưởng về đích (Cốt lõi)
        if reached_goal: return 100.0
        if robot.check_collision(obstacles): return -50.0

        # 2. Phần thưởng tiến độ
        progress_reward = 0
        if prev_distance is not None:
            progress_reward = (prev_distance - distance_to_goal) * 10.0
            # Phạt nặng hơn nếu đi ra xa
            if distance_to_goal > prev_distance:
                progress_reward -= 5.0
        
        # 3. Phạt cho mỗi bước đi (Tăng cường)
        step_penalty = -0.5

        # 4. Hình phạt góc nhìn (Angle Penalty)
        angle_penalty = 0
        robot_pos = np.array([robot.x, robot.y])
        goal_pos = np.array(self.goal)
        
        vec_to_goal = goal_pos - robot_pos
        last_move = robot.get_last_move() # Cần thêm hàm này vào class Robot
        
        if np.linalg.norm(last_move) > 0 and np.linalg.norm(vec_to_goal) > 0:
            cos_angle = np.dot(last_move, vec_to_goal) / (np.linalg.norm(last_move) * np.linalg.norm(vec_to_goal))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_diff = math.acos(cos_angle) # Góc lệch (radians)
            
            # Phạt dựa trên bình phương của góc lệch (khuyến khích đi thẳng)
            if angle_diff > 0.52: 
                angle_penalty = -2.0 * (angle_diff / math.pi)**2

        # 5. Hình phạt lặp lại và trì trệ (Giữ nguyên)
        position = (robot.grid_x, robot.grid_y)
        if position in self.position_history: self.position_history[position] += 1
        else: self.position_history[position] = 1
        self.position_history_list.append(position)
        if len(self.position_history_list) > self.position_memory_size:
            old = self.position_history_list.pop(0)
            if old in self.position_history:
                self.position_history[old] -= 1
                if self.position_history[old] <= 0: del self.position_history[old]
        
        repetition_penalty = -2.0 * (self.position_history.get(position, 1) - 1)
        
        stagnation_penalty = 0
        if len(self.position_history_list) > 20:
            if len(set(self.position_history_list[-20:])) <= 4:
                stagnation_penalty = -5.0

        return progress_reward + step_penalty + angle_penalty + repetition_penalty + stagnation_penalty
        
    def save_model(self):
        torch.save(self.q_network.state_dict(), self.model_path)
    def load_model(self):
        if os.path.exists(self.model_path):
            self.q_network.load_state_dict(torch.load(self.model_path, map_location=self.device))