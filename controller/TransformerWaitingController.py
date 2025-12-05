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
# 1. IMPROVED A* PLANNER (Lấy từ WaitingController)
# ==============================================================================
class AStarPlanner:
    def __init__(self, start, goal, obstacles, grid_width, grid_height, robot_radius):
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.width = grid_width
        self.height = grid_height
        self.resolution = max(1, int(robot_radius * 0.5))

    def plan(self):
        open_set = []
        heapq.heappush(open_set, (0, 0, self.start[0], self.start[1], [self.start]))
        visited = {}
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while open_set:
            f, g, cx, cy, path = heapq.heappop(open_set)

            if np.hypot(cx - self.goal[0], cy - self.goal[1]) <= self.resolution * 2.0:
                path.append(self.goal)
                return self.smooth_path(path)

            grid_pos = (int(cx // self.resolution), int(cy // self.resolution))
            if grid_pos in visited and visited[grid_pos] <= g: continue
            visited[grid_pos] = g

            for dx, dy in motions:
                nx, ny = cx + dx * self.resolution, cy + dy * self.resolution
                if not (0 <= nx <= self.width and 0 <= ny <= self.height): continue
                if not self._is_safe(nx, ny): continue

                move_cost = np.hypot(dx, dy)
                new_g = g + move_cost
                
                # --- LOGIC TỐI ƯU CỦA WAITING CONTROLLER ---
                h = np.hypot(nx - self.goal[0], ny - self.goal[1])
                # Cross product để ưu tiên đường thẳng
                dx1, dy1 = cx - self.goal[0], cy - self.goal[1]
                dx2, dy2 = self.start[0] - self.goal[0], self.start[1] - self.goal[1]
                cross = abs(dx1 * dy2 - dx2 * dy1)
                
                heuristic = h * 1.001 + cross * 0.001
                # -------------------------------------------

                new_path = list(path)
                new_path.append((nx, ny))
                heapq.heappush(open_set, (new_g + heuristic, new_g, nx, ny, new_path))
        return None

    def smooth_path(self, path):
        if len(path) <= 2: return path
        smoothed = [path[0]]
        cur_idx = 0
        while cur_idx < len(path) - 1:
            last_valid = cur_idx + 1
            check_range = min(len(path), cur_idx + 15)
            for i in range(check_range - 1, cur_idx, -1):
                if self._is_line_safe(path[cur_idx], path[i]):
                    last_valid = i
                    break
            smoothed.append(path[last_valid])
            cur_idx = last_valid
        return smoothed

    def _is_line_safe(self, start, end):
        x1, y1 = start
        x2, y2 = end
        dist = np.hypot(x2 - x1, y2 - y1)
        if dist == 0: return True
        steps = int(dist / (self.robot_radius * 0.5)) + 1
        for i in range(steps + 1):
            t = i / steps
            if not self._is_safe(x1 + (x2 - x1)*t, y1 + (y2 - y1)*t): return False
        return True

    def _is_safe(self, x, y):
        check_r = self.robot_radius * 1.1
        for obs in self.obstacles:
            r = pygame.Rect(obs.x - obs.width/2 - check_r, obs.y - obs.height/2 - check_r,
                            obs.width + 2*check_r, obs.height + 2*check_r)
            if r.collidepoint(x, y): return False
        return True

# ==============================================================================
# 2. WAITING RULE (Lấy từ WaitingController)
# ==============================================================================
class WaitingRule:
    def __init__(self, prediction_horizon=15):
        self.prediction_horizon = prediction_horizon

    def get_time_to_collision(self, robot, direction, dynamic_obstacles):
        rob_pos = np.array([robot.x, robot.y])
        rob_vel = np.array(direction) * robot.cell_size # Giả định tốc độ 1 cell/step
        safe_dist = robot.radius + 20 # Margin an toàn
        min_ttc = float('inf')

        for obs in dynamic_obstacles:
            obs_pos = np.array([obs.x, obs.y])
            obs_vel = np.array(obs.velocity) 
            for t in range(1, self.prediction_horizon + 1):
                f_rob = rob_pos + rob_vel * t
                f_obs = obs_pos + obs_vel * t
                if np.linalg.norm(f_rob - f_obs) < safe_dist:
                    if t < min_ttc: min_ttc = t
                    break
        return min_ttc if min_ttc != float('inf') else None

# ==============================================================================
# 3. TRANSFORMER MODEL (Giữ nguyên)
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
        self.feature = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))
    def forward(self, x):
        x = self.input_embed(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x[:, -1, :] 
        x = self.feature(x)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# ==============================================================================
# 4. MEMORY (Giữ nguyên)
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
# 5. TRANSFORMER WAITING CONTROLLER (MAIN)
# ==============================================================================
class TransformerWaitingController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=True, model_path="transformer_waiting.pth"):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
        self.sequence_length = 8
        self.state_dim = 36 # Vision(25) + Dist(1) + Ang(2) + Waypoint(2) + LastAct(1) + Dyn(5)
        self.action_dim = len(self.directions)
        
        gpu_id = find_free_gpu()
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        self.q_network = TransformerDuelingDQN(self.state_dim, self.action_dim, self.sequence_length).to(self.device)
        self.target_network = TransformerDuelingDQN(self.state_dim, self.action_dim, self.sequence_length).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        
        self.memory = PrioritizedReplayMemory(50000)
        
        # --- KẾ THỪA CÁC MODULE TỪ WAITING CONTROLLER ---
        self.waiting_rule = WaitingRule(prediction_horizon=15)
        self.known_static_obstacles = [] # Bộ nhớ map tĩnh
        self.known_ids = set()
        
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.padding_state = np.zeros(self.state_dim)
        
        # Internal states for A* and Stuck detection
        self.current_path = None
        self.path_index = 0
        self.last_action_idx = 0
        self.last_position = None
        self.stuck_counter = 0
        self.reversing_steps = 0
        
        self.gamma = 0.99
        self.epsilon = 1.0 if is_training else 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.target_update_freq = 500
        self.step_count = 0
        
        if not is_training: self.load_model()

    def _update_vision_memory(self, robot, all_obstacles):
        """Học map tĩnh như WaitingController"""
        found = False
        vision_sq = robot.vision ** 2
        for obs in all_obstacles:
            if not obs.static: continue
            if id(obs) in self.known_ids: continue
            if (obs.x - robot.x)**2 + (obs.y - robot.y)**2 <= vision_sq:
                self.known_static_obstacles.append(obs)
                self.known_ids.add(id(obs))
                found = True
        return found

    def _find_escape_direction(self, robot, obstacles):
        """Hàm thoát kẹt đơn giản"""
        dirs = list(self.directions)
        random.shuffle(dirs)
        for d in dirs:
            # Check nhanh va chạm tĩnh
            nx = robot.x + d[0]*robot.cell_size
            ny = robot.y + d[1]*robot.cell_size
            rect = pygame.Rect(nx-robot.radius, ny-robot.radius, robot.radius*2, robot.radius*2)
            collide = False
            for obs in obstacles:
                o_rect = pygame.Rect(obs.x-obs.width/2, obs.y-obs.height/2, obs.width, obs.height)
                if rect.colliderect(o_rect):
                    collide = True; break
            if not collide: return d
        return (0,0)

    def get_augmented_state(self, robot, obstacles):
        # 1. Update Map tĩnh & Check Path
        found_new = self._update_vision_memory(robot, obstacles)
        
        # Plan A* nếu chưa có đường hoặc phát hiện vật cản mới hoặc đã đi hết
        if self.current_path is None or found_new or self.path_index >= len(self.current_path):
            # Tính kích thước grid thực tế (pixels) cho A*
            gw_px = self.env_padding*2 + self.grid_width*self.cell_size
            gh_px = self.env_padding*2 + self.grid_height*self.cell_size
            planner = AStarPlanner((robot.x, robot.y), self.goal, self.known_static_obstacles, 
                                 gw_px, gh_px, robot.radius)
            self.current_path = planner.plan()
            self.path_index = 0

        # Lấy waypoint
        target_pt = self.goal
        if self.current_path:
            closest_dist = float('inf')
            closest_idx = self.path_index
            for i in range(self.path_index, min(len(self.current_path), self.path_index + 10)):
                px, py = self.current_path[i]
                d = math.hypot(px - robot.x, py - robot.y)
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i
            target_idx = min(closest_idx + 3, len(self.current_path) - 1)
            target_pt = self.current_path[target_idx]
            self.path_index = closest_idx

        # --- TẠO STATE VECTOR ---
        state_matrix, dist = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal)
        flat_vision = state_matrix.flatten() # 25
        
        # Goal & Waypoint Angles
        dx_g, dy_g = self.goal[0]-robot.x, self.goal[1]-robot.y
        ang_g = math.atan2(dy_g, dx_g)
        dx_w, dy_w = target_pt[0]-robot.x, target_pt[1]-robot.y
        ang_w = math.atan2(dy_w, dx_w)
        
        # Dynamic Obs Info
        moving_obs = robot.detect_moving_obstacles(obstacles)
        dyn_dist, dyn_sin, dyn_cos, dyn_vx, dyn_vy = 1.0, 0, 0, 0, 0
        if moving_obs:
            nearest = moving_obs[0]
            d, (dx, dy), vel = nearest['distance'], nearest['direction'], nearest['velocity']
            dyn_dist = min(d, robot.vision)/robot.vision
            a = math.atan2(dy, dx)
            dyn_sin, dyn_cos = math.sin(a), math.cos(a)
            dyn_vx, dyn_vy = vel[0]/5.0, vel[1]/5.0

        augmented = np.concatenate([
            flat_vision,
            [min(dist, 100.0)/100.0],
            [math.sin(ang_g), math.cos(ang_g)],
            [math.sin(ang_w), math.cos(ang_w)],
            [self.last_action_idx / self.action_dim],
            [dyn_dist, dyn_sin, dyn_cos, dyn_vx, dyn_vy]
        ])
        
        return augmented, target_pt

    def make_decision(self, robot, obstacles):
        # 1. Xử lý kẹt cứng (Stuck Handling từ WaitingController)
        if self.reversing_steps > 0:
            self.reversing_steps -= 1
            return self._find_escape_direction(robot, obstacles)

        robot_pos = (robot.x, robot.y)
        if self.last_position and np.linalg.norm(np.array(robot_pos) - np.array(self.last_position)) < 0.5:
             self.stuck_counter += 1
        else:
             self.stuck_counter = 0
        self.last_position = robot_pos
        
        if self.stuck_counter > 20: # Panic mode
            self.reversing_steps = 15
            self.current_path = None # Force replan
            self.stuck_counter = 0
            return self._find_escape_direction(robot, obstacles)

        # 2. Get State & Prepare Transformer
        state_vec, self.current_waypoint = self.get_augmented_state(robot, obstacles)
        self.sequence_buffer.append(state_vec)
        if len(self.sequence_buffer) < self.sequence_length:
            seq = [self.padding_state]*(self.sequence_length-len(self.sequence_buffer)) + list(self.sequence_buffer)
        else:
            seq = list(self.sequence_buffer)
        state_tensor = torch.FloatTensor(np.array(seq)).unsqueeze(0).to(self.device)

        # 3. MASKING THÔNG MINH (Kết hợp WaitingRule)
        # 1: Safe, 0: Dangerous
        mask = np.ones(self.action_dim)
        dynamic_obstacles = [o for o in obstacles if not o.static]
        
        for i, direction in enumerate(self.directions):
            # A. Check Static (Tường)
            nx = robot.x + direction[0]*robot.cell_size
            ny = robot.y + direction[1]*robot.cell_size
            rect = pygame.Rect(nx-robot.radius, ny-robot.radius, robot.radius*2, robot.radius*2)
            for obs in obstacles:
                if obs.static:
                    o_rect = pygame.Rect(obs.x-obs.width/2, obs.y-obs.height/2, obs.width, obs.height)
                    if rect.colliderect(o_rect):
                        mask[i] = 0; break
            if mask[i] == 0: continue

            # B. Check Dynamic (TTC) - Kỹ thuật WaitingRule
            ttc = self.waiting_rule.get_time_to_collision(robot, direction, dynamic_obstacles)
            if ttc is not None and ttc < 6: # Nếu va chạm trong < 6 bước
                mask[i] = 0 # Cấm đi hướng này
        
        # 4. CHỌN HÀNH ĐỘNG
        # Nếu tất cả các hướng di chuyển đều nguy hiểm (mask toàn 0), buộc phải ĐỨNG YÊN (0,0)
        # Giả sử trong self.directions không có (0,0), ta trả về (0,0) luôn và không qua NN
        if np.sum(mask) == 0:
            self.last_action_idx = 0 # Reset
            return (0, 0) # WAITING
        
        if self.is_training and random.random() < self.epsilon:
            valid_indices = [i for i, m in enumerate(mask) if m == 1]
            action_idx = random.choice(valid_indices) if valid_indices else 0
        else:
            with torch.no_grad():
                self.q_network.eval()
                q_values = self.q_network(state_tensor).cpu().numpy().squeeze()
                self.q_network.train()
                q_values[mask == 0] = -1e9
                action_idx = np.argmax(q_values)
                if q_values[action_idx] == -1e9: 
                    return (0, 0) # Fallback wait

        self.last_action_idx = action_idx
        return self.directions[action_idx]

    def store_experience(self, state, action_idx, reward, next_state, done):
        # (Giống logic buffer cũ)
        curr_seq_list = list(self.sequence_buffer)
        if len(curr_seq_list) < self.sequence_length:
            curr_seq = [self.padding_state]*(self.sequence_length-len(curr_seq_list)) + curr_seq_list
        else:
            curr_seq = curr_seq_list
            
        if not hasattr(self, 'temp_transition'): self.temp_transition = None
        
        if self.temp_transition:
            prev_seq, prev_a, prev_r = self.temp_transition
            self.memory.add((np.array(prev_seq), prev_a, prev_r, np.array(curr_seq), False))
            
        if done:
            next_seq_terminal = list(curr_seq)[1:] + [np.zeros(self.state_dim)]
            self.memory.add((np.array(curr_seq), action_idx, reward, np.array(next_seq_terminal), True))
            self.temp_transition = None
            self.sequence_buffer.clear()
            self.current_path = None
        else:
            self.temp_transition = (curr_seq, action_idx, reward)

    def calculate_reward(self, robot, obstacles, done, reached_goal, dist, prev_distance=None):
        reward = 0
        if reached_goal: return 100.0
        if done: return -50.0 
        
        # 1. Alignment Reward
        if hasattr(self, 'current_waypoint'):
            dx, dy = self.current_waypoint[0]-robot.x, self.current_waypoint[1]-robot.y
            move = robot.last_move_vector if robot.last_move_vector is not None else [0,0]
            if np.linalg.norm(move) > 0:
                mn = move / np.linalg.norm(move)
                tn = np.array([dx, dy]) / (np.linalg.norm([dx, dy]) + 1e-5)
                reward += np.dot(mn, tn) * 0.5 
        
        # 2. Dynamic Safety Reward (Khuyến khích đứng yên khi nguy hiểm)
        # Nếu robot đứng yên (move=0) nhưng có vật cản động ở rất gần -> Thưởng thông minh
        moving_obs = robot.detect_moving_obstacles(obstacles)
        if moving_obs and moving_obs[0]['distance'] < 30:
             if np.linalg.norm(robot.last_move_vector) < 0.1:
                 reward += 0.5 # Good waiting!
        
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