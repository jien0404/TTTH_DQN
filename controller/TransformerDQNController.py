import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import pygame
import heapq
from collections import deque
import os
from controller.Controller import Controller
from utils.gpu_utils import find_free_gpu

# ==============================================================================
# 1. CÁC MODULE PHỤ TRỢ (A* & SAFETY)
# ==============================================================================
class AStarPlanner:
    """Người dẫn đường: Tính toán đường đi ngắn nhất toàn cục"""
    def __init__(self, resolution, padding):
        self.resolution = resolution
        self.padding = padding

    def plan(self, start_pos, goal_pos, obstacles, grid_w, grid_h):
        start_node = (int((start_pos[0] - self.padding) // self.resolution),
                      int((start_pos[1] - self.padding) // self.resolution))
        goal_node = (int((goal_pos[0] - self.padding) // self.resolution),
                     int((goal_pos[1] - self.padding) // self.resolution))

        if start_node == goal_node: return [goal_pos]

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: math.hypot(start_node[0]-goal_node[0], start_node[1]-goal_node[1])}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal_node:
                return self._reconstruct_path(came_from, current, goal_pos)

            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < grid_w and 0 <= neighbor[1] < grid_h): continue
                
                # Check collision (chỉ check vật cản tĩnh cho A*)
                px = self.padding + (neighbor[0] + 0.5) * self.resolution
                py = self.padding + (neighbor[1] + 0.5) * self.resolution
                if self._is_colliding(px, py, obstacles): continue

                tentative_g = g_score[current] + math.hypot(dx, dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + math.hypot(neighbor[0]-goal_node[0], neighbor[1]-goal_node[1])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _is_colliding(self, x, y, obstacles):
        point_rect = pygame.Rect(x-2, y-2, 4, 4)
        for obs in obstacles:
            if obs.static:
                obs_rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
                if obs_rect.colliderect(point_rect): return True
        return False

    def _reconstruct_path(self, came_from, current, goal_pos):
        path = []
        while current in came_from:
            px = self.padding + (current[0] + 0.5) * self.resolution
            py = self.padding + (current[1] + 0.5) * self.resolution
            path.append((px, py))
            current = came_from[current]
        path.reverse()
        path.append(goal_pos)
        return path
    
    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.3, tolerance=0.00001):
        """Làm mượt path bằng gradient descent"""
        if len(path) < 3: return path
        
        new_path = np.array(path, dtype=float)
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                for j in range(2):
                    aux = new_path[i][j]
                    new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                    new_path[i][j] += weight_smooth * (new_path[i-1][j] + new_path[i+1][j] - 2*new_path[i][j])
                    change += abs(aux - new_path[i][j])
        return [tuple(p) for p in new_path]

class SafetyModule:
    """Lớp bảo vệ: Ngăn chặn hành động tự sát"""
    def __init__(self, cell_size, padding):
        self.cell_size = cell_size
        self.padding = padding

    def get_safe_mask(self, robot, obstacles, directions):
        mask = [True] * len(directions)
        robot_grid_pos = (robot.grid_x, robot.grid_y)
        
        for idx, (dx, dy) in enumerate(directions):
            next_gx = robot.grid_x + dx
            next_gy = robot.grid_y + dy
            
            # 1. Check Biên map (đơn giản hoá, check trong Controller chính kĩ hơn)
            # 2. Check Vật cản (Static + Dynamic)
            next_px = self.padding + (next_gx + 0.5) * self.cell_size
            next_py = self.padding + (next_gy + 0.5) * self.cell_size
            
            # Tạo rect giả lập robot
            robot_rect = pygame.Rect(next_px - robot.radius, next_py - robot.radius, 
                                     robot.radius*2, robot.radius*2)
            
            for obs in obstacles:
                obs_rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
                
                # Check va chạm tĩnh
                if robot_rect.colliderect(obs_rect):
                    mask[idx] = False
                    break
                
                # Check vùng nguy hiểm vật cản động (Dynamic Buffer)
                if not obs.static:
                    dist = math.hypot(obs.x - next_px, obs.y - next_py)
                    if dist < 30: # Vùng đệm an toàn 30px
                        mask[idx] = False
                        break
        return mask

# ==============================================================================
# 2. KIẾN TRÚC MẠNG TRANSFORMER (GIỮ NGUYÊN CORE, TỐI ƯU INPUT)
# ==============================================================================
class PositionalEncoding(nn.Module):
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
    def __init__(self, input_dim, output_dim, sequence_length=10, d_model=128, nhead=4, num_encoder_layers=2):
        super(TransformerDuelingDQN, self).__init__()
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.feature = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU()
        )
        self.value_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))

    def forward(self, x):
        x = self.input_embed(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x[:, -1, :] # Chỉ lấy token cuối cùng
        x = self.feature(x)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# ==============================================================================
# 3. PER (MEMORY)
# ==============================================================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = 0
    def add(self, p, data):
        idx = self.pending_idx + self.capacity - 1
        self.data[self.pending_idx] = data
        self.update(idx, p)
        self.pending_idx += 1
        if self.pending_idx >= self.capacity: self.pending_idx = 0
        if self.n_entries < self.capacity: self.n_entries += 1
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): return idx
        if s <= self.tree[left]: return self._retrieve(left, s)
        else: return self._retrieve(right, s - self.tree[left])
    def total(self): return self.tree[0]

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.max_priority = 1.0
    def add(self, experience):
        self.tree.add(self.max_priority ** self.alpha, experience)
    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + 0.001)
        for i in range(batch_size):
            (idx, p, data) = self.tree.get(random.uniform(segment * i, segment * (i + 1)))
            if data is not None:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)
        if not batch: return None, None, None
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        weights /= weights.max()
        return batch, idxs, weights
    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, (p + self.epsilon) ** self.alpha)
            self.max_priority = max(self.max_priority, p + self.epsilon)
    def __len__(self): return self.tree.n_entries

# ==============================================================================
# 4. ULTIMATE CONTROLLER
# ==============================================================================
class TransformerDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=True, model_path="transformer_dqn_model.pth"):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
        # --- Config ---
        self.sequence_length = 8 # Độ dài chuỗi nhớ
        
        # INPUT DIMENSION MỚI:
        # Vision(25) + Dist(1) + AngleGoal(2) + AngleWaypoint(2) + LastAction(1) = 31
        self.state_dim = 25 + 1 + 2 + 2 + 1 
        self.action_dim = len(self.directions)
        
        # Init components
        gpu_id = find_free_gpu()
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        self.q_network = TransformerDuelingDQN(self.state_dim, self.action_dim, self.sequence_length).to(self.device)
        self.target_network = TransformerDuelingDQN(self.state_dim, self.action_dim, self.sequence_length).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001) # Learning rate thấp cho ổn định
        
        self.memory = PrioritizedReplayMemory(50000)
        self.planner = AStarPlanner(cell_size, env_padding)
        self.safety = SafetyModule(cell_size, env_padding)
        
        # Buffers
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.padding_state = np.zeros(self.state_dim)
        self.current_path = []
        self.path_index = 0
        self.last_action_idx = 0 # Để đưa vào input (Smoothness)
        
        # RL params
        self.gamma = 0.99
        self.epsilon = 1.0 if is_training else 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.target_update_freq = 500
        self.step_count = 0
        
        if not is_training: self.load_model()

    def get_augmented_state(self, robot, obstacles):
        # 1. Vision & Dist (Cơ bản)
        state_matrix, dist = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal)
        flat_vision = state_matrix.flatten()
        
        # 2. Angle to Goal
        dx_g = self.goal[0] - robot.x
        dy_g = self.goal[1] - robot.y
        ang_g = math.atan2(dy_g, dx_g)
        
        # 3. Angle to A* Waypoint (Global Guidance)
        if not self.current_path or self.path_index >= len(self.current_path):
            self.current_path = self.planner.plan((robot.x, robot.y), self.goal, obstacles, self.grid_width, self.grid_height)
            self.path_index = 0
        
        # Logic bám waypoint
        target_pt = self.goal
        if self.current_path:
            # Smooth path nếu chưa smooth
            if not hasattr(self, '_smoothed_path') or self._path_version != id(self.current_path):
                self._smoothed_path = self.smooth_path(self.current_path)
                self._path_version = id(self.current_path)
            
            path_to_use = self._smoothed_path if hasattr(self, '_smoothed_path') else self.current_path
            
            # Tìm điểm gần nhất
            closest_dist = float('inf')
            closest_idx = self.path_index
            for i in range(self.path_index, min(len(path_to_use), self.path_index + 8)):
                px, py = path_to_use[i]
                d = math.hypot(px - robot.x, py - robot.y)
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i
            
            # === LOOKAHEAD ĐỘNG (thay vì cố định 2) ===
            robot_speed = math.hypot(robot.vx, robot.vy) if hasattr(robot, 'vx') else 5.0
            lookahead_steps = max(3, min(int(robot_speed / self.cell_size * 2), 6))
            target_idx = min(closest_idx + lookahead_steps, len(path_to_use) - 1)
            target_pt = path_to_use[target_idx]
            self.path_index = closest_idx

        dx_w = target_pt[0] - robot.x
        dy_w = target_pt[1] - robot.y
        ang_w = math.atan2(dy_w, dx_w)
        
        # 4. Normalize Dist
        norm_dist = min(dist, 100.0) / 100.0
        
        # 5. Last Action (giúp smooth đường đi)
        norm_last_action = self.last_action_idx / self.action_dim
        
        # Tổng hợp: 25 + 1 + 2 + 2 + 1 = 31
        augmented = np.concatenate([
            flat_vision, 
            [norm_dist], 
            [math.sin(ang_g), math.cos(ang_g)],
            [math.sin(ang_w), math.cos(ang_w)],
            [norm_last_action]
        ])
        return augmented, target_pt

    def make_decision(self, robot, obstacles):
        state_vec, self.current_waypoint = self.get_augmented_state(robot, obstacles)
        self.last_raw_state = state_vec # Lưu để dùng cho store_experience
        
        # Sequence Management
        self.sequence_buffer.append(state_vec)
        if len(self.sequence_buffer) < self.sequence_length:
            seq = [self.padding_state] * (self.sequence_length - len(self.sequence_buffer)) + list(self.sequence_buffer)
        else:
            seq = list(self.sequence_buffer)
        
        # Tensorize
        state_tensor = torch.FloatTensor(np.array(seq)).unsqueeze(0).to(self.device)
        
        # Safety Mask
        safe_mask = self.safety.get_safe_mask(robot, obstacles, self.directions)
        
        # Action Selection
        action_idx = 0
        if self.is_training and random.random() < self.epsilon:
            safe_indices = [i for i, safe in enumerate(safe_mask) if safe]
            if safe_indices: action_idx = random.choice(safe_indices)
            else: action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                self.q_network.eval()
                q_values = self.q_network(state_tensor).cpu().numpy().squeeze()
                self.q_network.train()
                
                # Masking: Gán -inf cho hành động nguy hiểm để không bao giờ chọn
                masked_q = np.copy(q_values)
                for i, safe in enumerate(safe_mask):
                    if not safe: masked_q[i] = -1e9

                # === THÊM MOMENTUM BONUS ===
                momentum_bonus = 0.3  # Tăng giá trị này nếu muốn mượt hơn
                if hasattr(self, 'last_action_idx'):
                    # Bonus cho action gần với last action
                    for i in range(len(masked_q)):
                        if masked_q[i] > -1e8:  # Action hợp lệ
                            dx_curr = self.directions[i][0]
                            dy_curr = self.directions[i][1]
                            dx_last = self.directions[self.last_action_idx][0]
                            dy_last = self.directions[self.last_action_idx][1]
                            
                            # Cosine similarity
                            similarity = (dx_curr*dx_last + dy_curr*dy_last) / max(
                                math.sqrt(dx_curr**2 + dy_curr**2) * math.sqrt(dx_last**2 + dy_last**2), 
                                1e-6
                            )
                            masked_q[i] += momentum_bonus * similarity
                
                action_idx = np.argmax(masked_q)
                # Fallback nếu kẹt cứng
                if masked_q[action_idx] == -1e9: action_idx = np.argmax(q_values)
        
        self.last_action_idx = action_idx
        return self.directions[action_idx]

    def store_experience(self, state, action_idx, reward, next_state, done):
        # Lưu ý: 'state' và 'next_state' từ main.py truyền vào là dạng cũ (vision, dist).
        # Ta cần dùng state augmented mà ta đã tính trong make_decision.
        
        # Xử lý: Ta lưu 'last_raw_state' vào bộ nhớ tạm.
        # Ở bước store này, ta chưa có next_augmented_state chính xác (vì robot vừa move xong).
        # Để đơn giản và hiệu quả: Ta lưu Transition (Current_Seq, Action, Reward, Done).
        # Next_Seq sẽ được tạo tự động khi lấy mẫu (sample) bằng cách dịch chuyển index.
        # Nhưng với ReplayBuffer đơn giản, ta cần lưu cả Next State.
        
        # Hack: Tính xấp xỉ next_augmented_state tại đây (với vị trí robot mới).
        # Cần truy cập lại obstacles. Do giới hạn API, ta giả sử robot đã ở vị trí mới.
        # Ta cần instance robot và obstacles. Main.py không truyền obstacles vào store_experience.
        # -> Giải pháp: Lưu tạm trong class ở make_decision và push vào buffer ở bước sau.
        pass # Logic phức tạp này được xử lý trong train() bằng cách lấy từ sequence buffer

        # Đơn giản hoá: Ta sẽ tự quản lý việc push vào memory bên trong make_decision hoặc
        # sửa lại flow. Nhưng để tuân thủ API:
        # Ta sẽ lưu trực tiếp state_sequence hiện tại.
        
        # Lấy current sequence
        curr_seq_list = list(self.sequence_buffer)
        if len(curr_seq_list) < self.sequence_length:
            curr_seq = [self.padding_state]*(self.sequence_length-len(curr_seq_list)) + curr_seq_list
        else:
            curr_seq = curr_seq_list
            
        # Tạo Next Sequence giả định (Shift trái + New State)
        # Vì ta không có next_augmented_state chính xác từ hàm này, ta dùng current_state của bước sau
        # làm next_state của bước này. Điều này yêu cầu bộ đệm 1 bước.
        
        if not hasattr(self, 'temp_transition'): self.temp_transition = None
        
        if self.temp_transition:
            prev_seq, prev_a, prev_r = self.temp_transition
            # curr_seq chính là next_seq của bước trước
            self.memory.add((np.array(prev_seq), prev_a, prev_r, np.array(curr_seq), False))
            
        if done:
            # Bước cuối
            next_seq_terminal = list(curr_seq)[1:] + [np.zeros(self.state_dim)] # Giả định
            self.memory.add((np.array(curr_seq), action_idx, reward, np.array(next_seq_terminal), True))
            self.temp_transition = None
            self.sequence_buffer.clear()
            self.current_path = [] # Reset path A*
        else:
            self.temp_transition = (curr_seq, action_idx, reward)

    def calculate_reward(self, robot, obstacles, done, reached_goal, dist, prev_distance=None):
        reward = 0
        if reached_goal: return 100.0
        if done: return -50.0 # Collision
        
        # 1. Progress Reward
        if prev_distance is not None:
            reward += (prev_distance - dist) * 15.0
            
        # 2. Alignment Reward (Đi đúng hướng A*)
        if hasattr(self, 'current_waypoint'):
            dx = self.current_waypoint[0] - robot.x
            dy = self.current_waypoint[1] - robot.y
            # Vector hướng di chuyển thực tế
            move = robot.last_move_vector if robot.last_move_vector is not None else [0,0]
            if np.linalg.norm(move) > 0:
                # Cosine similarity
                move_norm = move / np.linalg.norm(move)
                target_norm = np.array([dx, dy]) / (np.linalg.norm([dx, dy]) + 1e-5)
                alignment = np.dot(move_norm, target_norm)
                reward += alignment * 0.5 
        
        # 3. Step Penalty
        reward -= 0.05
        
        return reward

    def train(self):
        if len(self.memory) < self.batch_size: return
        batch, idxs, weights = self.memory.sample(self.batch_size)
        if not batch: return
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # DQN Logic
        curr_q = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        td_errors = torch.abs(curr_q - target_q).detach()
        loss = (weights * (curr_q - target_q).pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.memory.update_priorities(idxs, td_errors.cpu().numpy())
        
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
    def save_model(self): torch.save(self.q_network.state_dict(), self.model_path)
    def load_model(self):
        if os.path.exists(self.model_path):
            self.q_network.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.target_network.load_state_dict(self.q_network.state_dict())
            print("Model loaded.")
    def _save_model_implementation(self): pass
    def _load_model_implementation(self): pass