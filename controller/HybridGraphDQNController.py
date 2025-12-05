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
# 1. DUELING DQN (GIỮ NGUYÊN KHÔNG ĐỔI)
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
# 2. IMPROVED A* PLANNER (Học từ WaitingController)
# ==============================================================================
class AStarPlanner:
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
        heapq.heappush(open_set, (0, 0, start_node[0], start_node[1], start_node)) # f, g, x, y, curr
        came_from = {}
        g_score = {start_node: 0}
        
        # Lưu start/goal gốc để tính tie-breaker
        self.start_grid = start_node
        self.goal_grid = goal_node
        
        while open_set:
            # Pop phần tử có f thấp nhất
            f, g, cx, cy, current = heapq.heappop(open_set)
            current = (cx, cy)

            if current == goal_node:
                return self._reconstruct_path(came_from, current, goal_pos)

            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < grid_w and 0 <= neighbor[1] < grid_h): continue
                
                # Check collision tĩnh
                px = self.padding + (neighbor[0] + 0.5) * self.resolution
                py = self.padding + (neighbor[1] + 0.5) * self.resolution
                if self._is_colliding_static(px, py, obstacles): continue

                tentative_g = g_score[current] + math.hypot(dx, dy)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # --- KỸ THUẬT TIE-BREAKER (Làm đường thẳng hơn) ---
                    h = math.hypot(neighbor[0]-goal_node[0], neighbor[1]-goal_node[1])
                    
                    # Cross product: Ưu tiên node nằm trên đường thẳng nối Start-Goal
                    dx1 = neighbor[0] - goal_node[0]
                    dy1 = neighbor[1] - goal_node[1]
                    dx2 = self.start_grid[0] - goal_node[0]
                    dy2 = self.start_grid[1] - goal_node[1]
                    cross = abs(dx1 * dy2 - dx2 * dy1)
                    
                    heuristic = h * 1.001 + cross * 0.001
                    
                    f = tentative_g + heuristic
                    heapq.heappush(open_set, (f, tentative_g, neighbor[0], neighbor[1], neighbor))
                    
        return None 

    def _is_colliding_static(self, x, y, obstacles):
        margin = 2 # Margin nhỏ để A* linh hoạt
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
# 3. WAITING RULE (Module tính toán va chạm động)
# ==============================================================================
class WaitingRule:
    def __init__(self, prediction_horizon=10):
        self.prediction_horizon = prediction_horizon

    def get_time_to_collision(self, robot, direction, dynamic_obstacles):
        # Robot position (pixel)
        rob_pos = np.array([robot.x, robot.y])
        # Giả sử tốc độ robot là 1 cell/step (hoặc speed thực tế nếu có)
        rob_vel = np.array(direction) * robot.cell_size 
        
        # Radius an toàn (Robot + Obs + Buffer)
        safe_dist = robot.radius + 15

        min_ttc = float('inf')

        for obs in dynamic_obstacles:
            obs_pos = np.array([obs.x, obs.y])
            obs_vel = np.array(obs.velocity)

            # Dự đoán tương lai
            for t in range(1, self.prediction_horizon + 1):
                f_rob = rob_pos + rob_vel * t
                f_obs = obs_pos + obs_vel * t
                
                dist = np.linalg.norm(f_rob - f_obs)
                if dist < safe_dist:
                    if t < min_ttc: min_ttc = t
                    break 
        
        return min_ttc if min_ttc != float('inf') else None

# ==============================================================================
# 4. HYBRID CONTROLLER (MAIN)
# ==============================================================================
class HybridGraphDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=True, model_path="hybrid_model.pth"):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # --- INPUT STATE (GIỮ NGUYÊN 30 DIMENSIONS) ---
        self.state_dim = 30 
        self.action_dim = 8
        
        # --- INIT RL ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0002)
        self.memory = deque(maxlen=20000)
        
        # Modules phụ trợ
        self.planner = AStarPlanner(cell_size, env_padding)
        self.waiting_rule = WaitingRule(prediction_horizon=12) # Nhìn trước 12 bước
        
        # Internal State
        self.current_path = []
        self.path_idx = 0
        self.last_action_vec = (0, 0)
        self.target_pt = goal
        self.last_state_vec = None
        
        # Params
        self.batch_size = 64
        self.gamma = 0.98
        self.epsilon = 1.0 if is_training else 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.target_update_freq = 200
        self.step_count = 0
        
        if not is_training: self.load_model()

    def get_state_augmented(self, robot, obstacles):
        # 1. Update Path A*
        if not self.current_path or self.path_idx >= len(self.current_path):
            self.current_path = self.planner.plan((robot.x, robot.y), self.goal, obstacles, self.grid_width, self.grid_height)
            self.path_idx = 0
            
        # 2. Look-ahead Target
        target = self.goal
        if self.current_path:
            closest_dist = float('inf')
            closest_idx = self.path_idx
            search_range = min(len(self.current_path), self.path_idx + 15)
            for i in range(self.path_idx, search_range):
                px, py = self.current_path[i]
                d = math.hypot(px - robot.x, py - robot.y)
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i
            
            # Look ahead xa hơn một chút để mượt
            lookahead_idx = min(closest_idx + 4, len(self.current_path) - 1)
            target = self.current_path[lookahead_idx]
            self.path_idx = closest_idx 
            self.target_pt = target

        # 3. Construct State (30 dims)
        local_grid, _ = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal, vision_size=5)
        vision_flat = (local_grid == -1).astype(float).flatten() 
        
        dx = target[0] - robot.x
        dy = target[1] - robot.y
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        
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
        
        # --- ACTION MASKING NÂNG CAO (Safety Shield) ---
        mask = np.ones(self.action_dim)
        dynamic_obstacles = [o for o in obstacles if not o.static]
        
        for i, (dx, dy) in enumerate(self.directions):
            # 1. Check Tường (Static)
            nx = robot.grid_x + dx
            ny = robot.grid_y + dy
            if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                mask[i] = 0; continue
            
            px = self.env_padding + (nx + 0.5) * self.cell_size
            py = self.env_padding + (ny + 0.5) * self.cell_size
            rect = pygame.Rect(px-2, py-2, 4, 4)
            for obs in obstacles:
                if obs.static:
                    o_rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
                    if o_rect.colliderect(rect):
                        mask[i] = 0; break
            
            if mask[i] == 0: continue

            # 2. Check Vật cản động (Dynamic - Waiting Rule)
            ttc = self.waiting_rule.get_time_to_collision(robot, (dx, dy), dynamic_obstacles)
            if ttc is not None and ttc < 6: # Nếu sẽ va chạm trong < 6 bước
                mask[i] = 0 # Cấm đi hướng này
        
        # Fallback: Nếu bị vây kín (tất cả mask=0), thử đứng yên (0,0) nếu có trong action space
        # Nhưng ở đây action space là 8 hướng di chuyển. Robot bắt buộc phải chọn hướng ít tệ nhất.
        
        # Epsilon-Greedy
        if self.is_training and random.random() < self.epsilon:
            valid_indices = [i for i, m in enumerate(mask) if m == 1]
            if valid_indices: action_idx = random.choice(valid_indices)
            else: action_idx = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy().squeeze()
                # Phạt nặng các hành động bị cấm
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
        tx, ty = self.target_pt
        desired_dx = tx - robot.x
        desired_dy = ty - robot.y
        actual_dx, actual_dy = self.last_action_vec
        
        des_norm = math.hypot(desired_dx, desired_dy)
        act_norm = math.hypot(actual_dx, actual_dy)
        
        if des_norm > 0 and act_norm > 0:
            alignment = (desired_dx*actual_dx + desired_dy*actual_dy) / (des_norm * act_norm)
            reward += alignment * 0.5 
        
        # 2. Dynamic Safety Reward (Thưởng cho việc chờ đợi thông minh)
        # Nếu robot chọn hướng đi an toàn (không bị mask chặn) khi có vật cản gần
        moving_obs = robot.detect_moving_obstacles(obstacles)
        if moving_obs and moving_obs[0]['distance'] < 40:
             # Nếu robot vẫn sống sót khi vật cản ở gần
             reward += 0.1

        # 3. Penalty Step
        reward -= 0.05
        return reward

    def store_experience(self, state, action_idx, reward, next_state, done):
        if not hasattr(self, 'temp_transition'): self.temp_transition = None
        
        if self.temp_transition:
            s, a, r = self.temp_transition
            self.memory.append((s, a, r, self.last_state_vec, False))
            
        if done:
            self.memory.append((self.last_state_vec, action_idx, reward, np.zeros(self.state_dim), True))
            self.temp_transition = None
            self.current_path = [] 
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