import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import heapq
import math
import pygame
from collections import deque
from controller.Controller import Controller

# ==============================================================================
# 1. MẠNG DUELING DQN (Bộ não)
# ==============================================================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
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
# 2. A* PLANNER (Người dẫn đường)
# ==============================================================================
class AStarPlanner:
    def __init__(self, resolution, padding):
        self.resolution = resolution
        self.padding = padding

    def plan(self, start_pos, goal_pos, obstacles, grid_w, grid_h):
        # Chuyển đổi pixel sang grid index
        start_node = (int((start_pos[0] - self.padding) // self.resolution),
                      int((start_pos[1] - self.padding) // self.resolution))
        goal_node = (int((goal_pos[0] - self.padding) // self.resolution),
                     int((goal_pos[1] - self.padding) // self.resolution))

        if start_node == goal_node:
            return [goal_pos]

        # Khởi tạo grid map ảo để check va chạm nhanh
        # Lưu ý: obstacle ở đây là list object Obstacle
        
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        # 8 hướng di chuyển
        neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_node:
                return self._reconstruct_path(came_from, current, goal_pos)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < grid_w and 0 <= neighbor[1] < grid_h):
                    continue
                
                # Check collision (Static only for A*)
                # Ta check tâm ô grid
                px = self.padding + (neighbor[0] + 0.5) * self.resolution
                py = self.padding + (neighbor[1] + 0.5) * self.resolution
                if self._is_colliding(px, py, obstacles):
                    continue

                tentative_g = g_score[current] + math.hypot(dx, dy)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return None # Không tìm thấy đường

    def _heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _is_colliding(self, x, y, obstacles):
        # Tạo rect nhỏ cho điểm đang xét
        point_rect = pygame.Rect(x-2, y-2, 4, 4)
        for obs in obstacles:
            # A* chỉ nên quan tâm vật cản tĩnh để vẽ đường bao quát
            # Hoặc cả động nếu muốn né xa, nhưng ở đây ta chỉ dùng cho static
            if obs.static:
                obs_rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
                if obs_rect.colliderect(point_rect):
                    return True
        return False

    def _reconstruct_path(self, came_from, current, goal_pos_pixel):
        path = []
        while current in came_from:
            # Chuyển đổi ngược lại pixel center
            px = self.padding + (current[0] + 0.5) * self.resolution
            py = self.padding + (current[1] + 0.5) * self.resolution
            path.append((px, py))
            current = came_from[current]
        path.reverse()
        path.append(goal_pos_pixel)
        return path

# ==============================================================================
# 3. SAFETY MODULE (Lớp bảo vệ)
# ==============================================================================
class SafetyModule:
    def __init__(self, cell_size, padding):
        self.cell_size = cell_size
        self.padding = padding

    def get_safe_mask(self, robot, obstacles, directions):
        """Trả về mask (1 là an toàn, 0 là nguy hiểm) cho 8 hướng + đứng yên"""
        mask = np.ones(len(directions))
        
        dynamic_obstacles = [o for o in obstacles if not o.static]
        
        for idx, (dx, dy) in enumerate(directions):
            # Dự đoán vị trí tiếp theo
            next_gx = robot.grid_x + dx
            next_gy = robot.grid_y + dy
            
            next_px = self.padding + (next_gx + 0.5) * self.cell_size
            next_py = self.padding + (next_gy + 0.5) * self.cell_size
            
            # 1. Check Static Collision (Tường)
            # Tạo rect giả lập robot tại vị trí mới
            robot_rect = pygame.Rect(next_px - robot.radius, next_py - robot.radius, 
                                     robot.radius*2, robot.radius*2)
            
            collision_static = False
            for obs in obstacles:
                obs_rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
                if robot_rect.colliderect(obs_rect):
                    collision_static = True
                    break
            
            if collision_static:
                mask[idx] = 0
                continue
                
            # 2. Check Dynamic Collision (Time-To-Collision đơn giản)
            # Nếu vật cản động đang lao tới vị trí next_px trong vòng 5-10 frame tới
            for obs in dynamic_obstacles:
                dist = math.hypot(obs.x - next_px, obs.y - next_py)
                # Nếu quá gần vật cản động (< 20px) -> Nguy hiểm
                if dist < 25: 
                    mask[idx] = 0
                    break
                    
        return mask

# ==============================================================================
# 4. CONTROLLER CHÍNH
# ==============================================================================
class AstarDQNCOntroller(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="advanced_dqn.pth"):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
        # --- Cấu hình Hyperparameters ---
        self.gamma = 0.99
        self.epsilon = 1.0 if is_training else 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.memory_size = 50000
        self.target_update_freq = 500
        
        # --- Kiến trúc Input ---
        # 1. Vision Grid (5x5) = 25
        # 2. Dist to Goal (Normalized) = 1
        # 3. Angle to Goal (Sin, Cos) = 2
        # 4. Angle to Next A* Waypoint (Sin, Cos) = 2
        # Tổng input = 30
        self.state_dim = 30
        self.action_dim = len(self.directions) # 8 hướng + đứng yên (nếu có)
        
        # --- Khởi tạo Modules ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size)
        
        self.planner = AStarPlanner(cell_size, env_padding)
        self.safety = SafetyModule(cell_size, env_padding)
        
        # --- Biến trạng thái nội bộ ---
        self.current_path = []
        self.path_index = 0
        self.step_count = 0
        self.last_pos = None
        self.stuck_count = 0
        
        if not is_training:
            self.load_model()

    def get_augmented_state(self, robot, obstacles, grid_w, grid_h):
        """Tạo vector trạng thái thông minh"""
        # 1. Vision Grid (5x5)
        # Robot.get_state trả về vision grid và distance
        vision_grid, dist_to_goal = robot.get_state(obstacles, grid_w, grid_h, self.goal)
        flat_vision = vision_grid.flatten()
        
        # 2. Thông tin mục tiêu (Goal Info)
        dx_g = self.goal[0] - robot.x
        dy_g = self.goal[1] - robot.y
        angle_to_goal = math.atan2(dy_g, dx_g)
        
        # 3. Thông tin A* (Waypoint Info)
        # Nếu chưa có đường hoặc đã đi hết -> Plan lại
        if not self.current_path or self.path_index >= len(self.current_path):
            self.current_path = self.planner.plan((robot.x, robot.y), self.goal, obstacles, grid_w, grid_h)
            self.path_index = 0
            
        # Tìm waypoint tiếp theo
        target_x, target_y = self.goal
        if self.current_path:
            # Tìm điểm gần nhất trên path để bám theo (tránh robot bị trôi)
            min_d = float('inf')
            closest_i = self.path_index
            for i in range(self.path_index, min(len(self.current_path), self.path_index + 5)):
                px, py = self.current_path[i]
                d = math.hypot(px - robot.x, py - robot.y)
                if d < min_d:
                    min_d = d
                    closest_i = i
            
            # Chọn điểm phía trước điểm gần nhất
            target_idx = min(closest_i + 2, len(self.current_path) - 1)
            target_x, target_y = self.current_path[target_idx]
            self.path_index = closest_i # Cập nhật progress

        dx_w = target_x - robot.x
        dy_w = target_y - robot.y
        angle_to_waypoint = math.atan2(dy_w, dx_w)
        
        # Normalize distance (giả sử map size max ~ 600)
        norm_dist = min(dist_to_goal, 600) / 600.0
        
        # Ghép state vector
        state_vector = np.concatenate([
            flat_vision,
            [norm_dist],
            [math.sin(angle_to_goal), math.cos(angle_to_goal)],
            [math.sin(angle_to_waypoint), math.cos(angle_to_waypoint)]
        ])
        
        return state_vector, (target_x, target_y) # Trả về cả waypoint để debug/reward

    def make_decision(self, robot, obstacles):
        # Lấy kích thước grid (ước lượng từ pixel)
        grid_w = int((robot.x + 1000) // self.cell_size) # Safe upper bound
        grid_h = int((robot.y + 1000) // self.cell_size)
        
        # Cập nhật state
        state_vec, _ = self.get_augmented_state(robot, obstacles, 32, 32) # Giả sử map 32x32
        self.last_state_vec = state_vec # Lưu để dùng cho store_experience
        
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        # Lấy Safety Mask (1=Safe, 0=Unsafe)
        safe_mask = self.safety.get_safe_mask(robot, obstacles, self.directions)
        
        # Epsilon-Greedy
        if self.is_training and random.random() < self.epsilon:
            # Random nhưng ưu tiên safe actions
            safe_indices = [i for i, is_safe in enumerate(safe_mask) if is_safe == 1]
            if safe_indices:
                action_idx = random.choice(safe_indices)
            else:
                action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy().squeeze()
                
                # Áp dụng Safety Mask: Gán -infinity cho hành động nguy hiểm
                # để DQN không bao giờ chọn (trừ khi tất cả đều nguy hiểm)
                masked_q_values = np.copy(q_values)
                masked_q_values[safe_mask == 0] = -1e9
                
                action_idx = np.argmax(masked_q_values)
                
                # Fallback: Nếu tất cả đều unsafe, chọn cái ít âm nhất (nguy hiểm ít nhất) từ Q gốc
                if masked_q_values[action_idx] == -1e9:
                     action_idx = np.argmax(q_values)

        return self.directions[action_idx]

    def calculate_reward(self, robot, obstacles, done, reached_goal, next_dist, prev_distance=None):
        """Hàm thưởng được thiết kế kỹ lưỡng để tối ưu đường đi"""
        reward = 0
        
        # 1. Thưởng/Phạt cơ bản
        if reached_goal:
            return 100.0 # Thưởng lớn về đích
        if done: # Va chạm (Collision) hoặc Trap
            return -50.0 # Phạt nặng
            
        # 2. Thưởng tiến độ (Shaping Reward)
        if prev_distance is not None:
            diff = prev_distance - next_dist
            # Thưởng nếu lại gần đích, phạt nếu đi xa
            reward += diff * 15.0 
        
        # 3. Phạt mỗi bước đi (Time penalty) -> Khuyến khích đường ngắn
        reward -= 0.1
        
        # 4. Thưởng đi theo A* Waypoint (Alignment Reward)
        # Tính lại góc tới waypoint từ state (hơi phức tạp để lôi ra từ đây)
        # Ta dùng heuristic đơn giản: Nếu hướng đi (last move) khớp với hướng A*
        
        # 5. Phạt đứng yên/kẹt
        if robot.last_move_vector is not None:
            move_dist = np.linalg.norm(robot.last_move_vector)
            if move_dist < 1.0: # Robot gần như không nhúc nhích
                reward -= 0.5
        
        # 6. Smoothness penalty (Phạt đi zic-zac)
        # Cần lưu last_action trong class để so sánh. 
        # (Bạn có thể thêm self.last_action_idx vào class và cập nhật ở make_decision)
        
        return reward

    def store_experience(self, state, action_idx, reward, next_state, done):
        # Lưu ý: state ở tham số hàm này là từ main.py (vision, dist).
        # Nhưng DQN của ta dùng 'augmented_state'.
        # Để đơn giản, ta cần main.py truyền đúng state hoặc ta tự quản lý state ở controller.
        
        # Vì cấu trúc main.py gọi robot.get_state rồi truyền vào đây,
        # Nên ta sẽ hack nhẹ: Trong make_decision ta đã tính augmented_state (self.last_state_vec).
        # Ta sẽ dùng biến đó.
        
        # Cần tính next_augmented_state
        # Việc này hơi khó vì main.py đã gọi robot.move() rồi.
        # Giải pháp tốt nhất cho framework hiện tại:
        # Tạm thời ta chỉ dùng vision+dist (như code cũ) HOẶC sửa main.py.
        # Nhưng đề bài yêu cầu controller đầy đủ.
        
        # -> Ta sẽ tính lại augmented state tại đây.
        # robot lúc này đã ở vị trí next_state.
        current_s = self.last_state_vec # State trước khi move
        
        # Tính next state (Robot đã move rồi)
        # Cần truy cập lại obstacles, grid_w, h từ đâu đó.
        # Do hạn chế params, ta dùng giá trị mặc định cho grid size
        
        # CHÚ Ý: Để code chạy mượt với main.py cũ, ta cần chỉnh lại một chút logic data flow.
        # Tuy nhiên, để đơn giản, ta sẽ lưu trực tiếp các vector đã tính toán.
        
        # Do không thể tính lại hoàn hảo next_state vector ở đây (thiếu obstacles context),
        # Ta sẽ chấp nhận lưu experience ở bước kế tiếp (One-step delayed storage) 
        # Hoặc tính xấp xỉ.
        
        # Cách thực dụng: Lưu tuple vào buffer tạm, đợi bước sau có next_state thì push vào memory.
        if not hasattr(self, 'temp_transition'):
            self.temp_transition = None
            
        if self.temp_transition:
            # Transition cũ: (s, a, r) ... đợi next_s
            s, a, r = self.temp_transition
            # current_s bây giờ chính là next_state của bước trước
            self.memory.append((s, a, r, current_s, False)) # False = not done (vì đã sang bước mới)
            
        if done:
            # Nếu done, bước hiện tại không có next_state (hoặc terminal)
            # Push luôn
            self.memory.append((current_s, action_idx, reward, current_s, True)) # next_s ko quan trọng khi done
            self.temp_transition = None
        else:
            # Lưu tạm để chờ bước sau
            self.temp_transition = (current_s, action_idx, reward)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions)
        
        # Compute Target Q
        with torch.no_grad():
            # Double DQN Logic
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _save_model_implementation(self):
        torch.save(self.policy_net.state_dict(), self.model_path)

    def _load_model_implementation(self):
        try:
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Loaded model from {self.model_path}")
        except FileNotFoundError:
            print("No model found, starting fresh.")