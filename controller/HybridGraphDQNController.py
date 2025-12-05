import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import heapq
import pygame
from collections import deque
import os
from controller.Controller import Controller

# ==============================================================================
# 1. DUELING DQN (Bộ não xử lý tình huống động)
# ==============================================================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# ==============================================================================
# 2. A* PLANNER (GPS chỉ đường tĩnh)
# ==============================================================================
class AStarPlanner:
    def __init__(self, resolution, padding):
        self.resolution = resolution
        self.padding = padding

    def plan(self, start_pos, goal_pos, obstacles, grid_w, grid_h):
        # Chuyển đổi pixel sang grid
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

            # 8 hướng
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check biên
                if not (0 <= neighbor[0] < grid_w and 0 <= neighbor[1] < grid_h): continue
                
                # Check va chạm tĩnh (Chỉ check tường đen)
                px = self.padding + (neighbor[0] + 0.5) * self.resolution
                py = self.padding + (neighbor[1] + 0.5) * self.resolution
                if self._is_colliding_static(px, py, obstacles): continue

                tentative_g = g_score[current] + math.hypot(dx, dy)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + math.hypot(neighbor[0]-goal_node[0], neighbor[1]-goal_node[1])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None 

    def _is_colliding_static(self, x, y, obstacles):
        # Check an toàn với margin lớn hơn một chút để A* không đi quá sát tường
        margin = 1 
        point_rect = pygame.Rect(x-margin, y-margin, margin*2, margin*2)
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

# ==============================================================================
# 3. HYBRID CONTROLLER
# ==============================================================================
class HybridGraphDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=True, model_path="hybrid_model.pth"):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # --- INPUT STATE ---
        # 1. Vision 5x5 flattened (25)
        # 2. Normalized Dist to Lookahead Point (1)
        # 3. Sin/Cos Angle to Lookahead Point (2)
        # 4. Last Action Velocity (2)
        # Total = 30
        self.state_dim = 30 
        self.action_dim = 8
        
        # --- INIT RL ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0002)
        self.memory = deque(maxlen=20000)
        
        self.planner = AStarPlanner(cell_size, env_padding)
        
        # --- Internal State ---
        self.current_path = []
        self.path_idx = 0
        self.last_action_vec = (0, 0)
        self.target_pt = goal
        self.last_state_vec = None # Để lưu experience
        
        # --- Params ---
        self.batch_size = 64
        self.gamma = 0.98
        self.epsilon = 1.0 if is_training else 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.target_update_freq = 200
        self.step_count = 0
        
        if not is_training: self.load_model()

    def get_state_augmented(self, robot, obstacles):
        # 1. Update Path A* nếu chưa có hoặc đã đi hết
        if not self.current_path or self.path_idx >= len(self.current_path):
            self.current_path = self.planner.plan((robot.x, robot.y), self.goal, obstacles, self.grid_width, self.grid_height)
            self.path_idx = 0
            
        # 2. Look-ahead Logic (Nhìn xa trông rộng)
        # Tìm điểm xa nhất trên đường A* mà robot có thể nhắm tới để làm mượt đường đi
        target = self.goal
        if self.current_path:
            closest_dist = float('inf')
            closest_idx = self.path_idx
            
            # Tìm điểm gần nhất trên path để sync lại vị trí robot
            search_range = min(len(self.current_path), self.path_idx + 15)
            for i in range(self.path_idx, search_range):
                px, py = self.current_path[i]
                d = math.hypot(px - robot.x, py - robot.y)
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i
            
            # Look ahead: Nhắm vào điểm cách điểm gần nhất khoảng 4-5 node
            # Điều này giúp robot cắt cua mượt hơn thay vì đi vuông góc
            lookahead_idx = min(closest_idx + 4, len(self.current_path) - 1)
            target = self.current_path[lookahead_idx]
            self.path_idx = closest_idx 
            self.target_pt = target

        # 3. Construct State
        # A. Vision (5x5)
        local_grid, _ = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal, vision_size=5)
        vision_flat = (local_grid == -1).astype(float).flatten() 
        
        # B. Info to Target
        dx = target[0] - robot.x
        dy = target[1] - robot.y
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        
        # C. Last Action
        ldx, ldy = self.last_action_vec
        
        state_vec = np.concatenate([
            vision_flat,
            [min(dist, 200) / 200.0],
            [math.sin(angle), math.cos(angle)],
            [ldx, ldy]
        ])
        
        return state_vec

    def make_decision(self, robot, obstacles):
        state_vec = self.get_state_augmented(robot, obstacles)
        self.last_state_vec = state_vec 
        
        # --- ACTION MASKING (KỸ THUẬT QUAN TRỌNG) ---
        # Chỉ cho phép chọn các hướng không đâm vào tường TĨNH
        # Giúp model hội tụ cực nhanh vì không cần học lại các tai nạn ngớ ngẩn
        mask = np.ones(self.action_dim)
        for i, (dx, dy) in enumerate(self.directions):
            nx = robot.grid_x + dx
            ny = robot.grid_y + dy
            # Check biên
            if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                mask[i] = 0; continue
                
            # Check tường tĩnh nhanh
            px = self.env_padding + (nx + 0.5) * self.cell_size
            py = self.env_padding + (ny + 0.5) * self.cell_size
            # Tạo rect nhỏ kiểm tra va chạm
            rect = pygame.Rect(px-2, py-2, 4, 4)
            
            # Lưu ý: Cần loop qua obstacles để check static
            for obs in obstacles:
                if obs.static:
                    o_rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
                    if o_rect.colliderect(rect):
                        mask[i] = 0
                        break
        
        # Epsilon-Greedy với Mask
        if self.is_training and random.random() < self.epsilon:
            valid_indices = [i for i, m in enumerate(mask) if m == 1]
            if valid_indices: action_idx = random.choice(valid_indices)
            else: action_idx = 0 # Fallback (bị vây kín)
        else:
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy().squeeze()
                # Phạt nặng các hành động bị mask
                q_values[mask == 0] = -1e9
                action_idx = np.argmax(q_values)
                
        self.last_action_idx = action_idx
        self.last_action_vec = self.directions[action_idx]
        return self.directions[action_idx]

    def calculate_reward(self, robot, obstacles, done, reached_goal, dist, prev_distance=None):
        reward = 0
        if reached_goal: return 100.0
        if done: return -50.0 
        
        # 1. Alignment Reward (Thưởng đi đúng hướng theo A*)
        # Đây là bí quyết giúp robot học nhanh: Thưởng khi vector di chuyển trùng với vector hướng về target
        tx, ty = self.target_pt
        desired_dx = tx - robot.x
        desired_dy = ty - robot.y
        
        actual_dx, actual_dy = self.last_action_vec
        
        # Tính Cosine Similarity
        des_norm = math.hypot(desired_dx, desired_dy)
        act_norm = math.hypot(actual_dx, actual_dy)
        
        if des_norm > 0 and act_norm > 0:
            alignment = (desired_dx*actual_dx + desired_dy*actual_dy) / (des_norm * act_norm)
            reward += alignment * 0.5 
            
        # 2. Time Penalty (Khuyến khích đi nhanh)
        reward -= 0.05
        
        return reward

    def store_experience(self, state, action_idx, reward, next_state, done):
        # Lưu experience vào buffer. Do cấu trúc main.py không trả về augmented next_state,
        # ta dùng kỹ thuật lưu state t và coi next_state là state t+1 ở bước sau.
        # Ở đây ta implement đơn giản: lưu last_state_vec
        
        if not hasattr(self, 'temp_transition'): self.temp_transition = None
        
        if self.temp_transition:
            s, a, r = self.temp_transition
            # self.last_state_vec hiện tại chính là next_state của bước trước
            self.memory.append((s, a, r, self.last_state_vec, False))
            
        if done:
            self.memory.append((self.last_state_vec, action_idx, reward, np.zeros(self.state_dim), True))
            self.temp_transition = None
            self.current_path = [] # Reset path khi hết episode
        else:
            self.temp_transition = (self.last_state_vec, action_idx, reward)

    def train(self):
        if len(self.memory) < self.batch_size: return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        curr_q = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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