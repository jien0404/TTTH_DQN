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
# 1. DUELING DQN (MẠNG NƠ-RON QUYẾT ĐỊNH)
# ==============================================================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Mạng đơn giản nhưng sâu để xử lý logic
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
# 2. A* PLANNER (GPS DẪN ĐƯỜNG)
# ==============================================================================
class AStarPlanner:
    def __init__(self, resolution, padding):
        self.resolution = resolution
        self.padding = padding

    def plan(self, start_pos, goal_pos, obstacles, grid_w, grid_h):
        # Chuyển đổi toạ độ thực sang toạ độ Grid
        start_node = (int((start_pos[0] - self.padding) // self.resolution),
                      int((start_pos[1] - self.padding) // self.resolution))
        goal_node = (int((goal_pos[0] - self.padding) // self.resolution),
                     int((goal_pos[1] - self.padding) // self.resolution))

        if start_node == goal_node: return [goal_pos]

        # Chuẩn bị A*
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        # Tạo bản đồ tĩnh dạng grid để truy xuất nhanh (cache)
        # Lưu ý: Hàm này giả định obstacles không thay đổi vị trí (Static only)
        
        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_node:
                return self._reconstruct_path(came_from, current, goal_pos)

            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check biên
                if not (0 <= neighbor[0] < grid_w and 0 <= neighbor[1] < grid_h): continue
                
                # Check va chạm tĩnh (Tường đen)
                px = self.padding + (neighbor[0] + 0.5) * self.resolution
                py = self.padding + (neighbor[1] + 0.5) * self.resolution
                if self._is_colliding_static(px, py, obstacles): continue

                tentative_g = g_score[current] + math.hypot(dx, dy)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return None 

    def _heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _is_colliding_static(self, x, y, obstacles):
        # Chỉ check vật cản tĩnh (Static = True)
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

# ==============================================================================
# 3. HYBRID CONTROLLER (MAIN)
# ==============================================================================
class HybridGraphDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=True, model_path="hybrid_model.pth"):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # --- Config Input ---
        # 1. Local Vision (5x5 flatten) = 25
        # 2. Distance to Target Waypoint = 1
        # 3. Angle to Target Waypoint (sin, cos) = 2
        # 4. Current Velocity (Last Action) = 2
        # Total = 30
        self.state_dim = 30 
        self.action_dim = 8 # 8 hướng
        
        # --- Components ---
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
        
        # --- RL Params ---
        self.batch_size = 64
        self.gamma = 0.98
        self.epsilon = 1.0 if is_training else 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.target_update_freq = 200
        self.step_count = 0
        
        if not is_training: self.load_model()

    def get_state_augmented(self, robot, obstacles):
        # 1. Update Path (A*) nếu cần
        if not self.current_path or self.path_idx >= len(self.current_path):
            self.current_path = self.planner.plan((robot.x, robot.y), self.goal, obstacles, self.grid_width, self.grid_height)
            self.path_idx = 0
            
        # 2. Tính toán "Look-ahead Target" (Điểm nhắm tới)
        # Thay vì nhắm điểm kế tiếp, ta tìm điểm xa nhất trên path mà mắt thường nhìn thấy (Line of Sight)
        # Điều này giúp làm mượt đường đi (Straight line shortcut)
        target = self.goal
        if self.current_path:
            # Tìm điểm gần nhất trên path
            closest_dist = float('inf')
            closest_idx = self.path_idx
            for i in range(self.path_idx, min(len(self.current_path), self.path_idx + 10)):
                px, py = self.current_path[i]
                d = math.hypot(px - robot.x, py - robot.y)
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i
            
            # Look ahead logic: Chọn điểm phía trước điểm gần nhất khoảng 3-4 bước
            lookahead_idx = min(closest_idx + 3, len(self.current_path) - 1)
            target = self.current_path[lookahead_idx]
            self.path_idx = closest_idx # Cập nhật tiến độ
            self.target_pt = target

        # 3. Tạo State Vector
        # A. Local Vision (5x5 grid) - Chỉ quan tâm vật cản ĐỘNG và Tĩnh ở gần
        local_grid, _ = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal, vision_size=5)
        # get_state trả về -1 là tường, 0 là đường. Ta normalize về [0, 1] (1 là có vật cản)
        vision_flat = (local_grid == -1).astype(float).flatten() 
        
        # B. Relative Info to Target
        dx = target[0] - robot.x
        dy = target[1] - robot.y
        dist_to_target = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        
        # C. Last Action (Inertia)
        last_dx, last_dy = self.last_action_vec
        
        state_vec = np.concatenate([
            vision_flat,
            [min(dist_to_target, 200) / 200.0], # Normalize dist
            [math.sin(angle_to_target), math.cos(angle_to_target)],
            [last_dx, last_dy]
        ])
        
        return state_vec, dist_to_target

    def make_decision(self, robot, obstacles):
        state_vec, dist_to_target = self.get_state_augmented(robot, obstacles)
        self.last_state_vec = state_vec # Lưu để train
        
        # Action Masking: Xác định hướng đi nào bị tường chặn
        # Mask = 1 (Đi được), 0 (Đâm tường)
        mask = np.ones(self.action_dim)
        for i, (dx, dy) in enumerate(self.directions):
            nx = robot.grid_x + dx
            ny = robot.grid_y + dy
            # Check biên
            if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                mask[i] = 0
                continue
            # Check tường tĩnh (Dùng hàm trong Robot hoặc tự check)
            # Ở đây ta check nhanh bằng cách loop obstacles tĩnh
            # (Hoặc tối ưu hơn là dùng grid map đã cache, nhưng loop cũng ok với số lượng ít)
            n_px = self.env_padding + (nx + 0.5) * self.cell_size
            n_py = self.env_padding + (ny + 0.5) * self.cell_size
            rect = pygame.Rect(n_px - 2, n_py - 2, 4, 4)
            for obs in obstacles:
                if obs.static:
                    o_rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
                    if o_rect.colliderect(rect):
                        mask[i] = 0
                        break
        
        # Epsilon Greedy với Mask
        if self.is_training and random.random() < self.epsilon:
            valid_indices = [i for i, m in enumerate(mask) if m == 1]
            if valid_indices: action_idx = random.choice(valid_indices)
            else: action_idx = 0 # Fallback
        else:
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy().squeeze()
                # Apply Mask: Gán giá trị cực thấp cho hành động bị cấm
                q_values[mask == 0] = -1e9
                action_idx = np.argmax(q_values)
                
        self.last_action_idx = action_idx
        self.last_action_vec = self.directions[action_idx]
        return self.directions[action_idx]

    def calculate_reward(self, robot, obstacles, done, reached_goal, dist, prev_distance=None):
        # Lưu ý: 'dist' ở đây là dist tới GOAL cuối cùng (từ main.py truyền vào)
        # Ta cũng nên tính dist tới TARGET (waypoint) để thưởng cục bộ
        
        reward = 0
        
        if reached_goal: return 100.0
        if done: return -50.0 # Collision
        
        # 1. Alignment Reward (Thưởng đi đúng hướng về Waypoint)
        # Tính lại vector tới waypoint (lấy từ self.target_pt đã tính ở get_state)
        tx, ty = self.target_pt
        desired_dx = tx - robot.x
        desired_dy = ty - robot.y
        
        actual_dx, actual_dy = self.last_action_vec
        
        if np.linalg.norm([desired_dx, desired_dy]) > 0:
            # Cosine similarity
            v1 = np.array([desired_dx, desired_dy]) / np.linalg.norm([desired_dx, desired_dy])
            v2 = np.array([actual_dx, actual_dy])
            if np.linalg.norm(v2) > 0: v2 = v2 / np.linalg.norm(v2)
            
            alignment = np.dot(v1, v2)
            reward += alignment * 0.5 # Thưởng 0.5 nếu đi đúng hướng
            
        # 2. Penalty Step
        reward -= 0.05
        
        # 3. Penalty Stagnation (Đứng yên)
        if actual_dx == 0 and actual_dy == 0:
            reward -= 0.2
            
        return reward

    def store_experience(self, state, action_idx, reward, next_state, done):
        # Hack nhẹ để tạo next_state vector vì main.py không trả về augmented state sau khi move
        # Ta dùng kỹ thuật "1-step delay buffer" hoặc tính xấp xỉ.
        # Ở đây ta dùng xấp xỉ đơn giản: State sau coi như giống state trước nhưng thay đổi vị trí
        # (Để code ngắn gọn, ta lưu trực tiếp transition raw và xử lý sau hoặc chấp nhận sai số nhỏ ở next_state visual)
        
        # Để chính xác, ta cần:
        # State lưu vào buffer PHẢI là augmented state (30 dims)
        # self.last_state_vec là state t.
        
        # Ta tạo một bộ đệm đơn giản:
        if not hasattr(self, 'temp_exp'): self.temp_exp = None
        
        if self.temp_exp:
            # s_t, a_t, r_t -> s_{t+1} (là self.last_state_vec hiện tại)
            s, a, r = self.temp_exp
            self.memory.append((s, a, r, self.last_state_vec, False))
            
        if done:
            # Terminal state -> next state là chuỗi 0
            self.memory.append((self.last_state_vec, action_idx, reward, np.zeros(self.state_dim), True))
            self.temp_exp = None
            self.current_path = [] # Reset A* path
        else:
            self.temp_exp = (self.last_state_vec, action_idx, reward)

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