# controller/WaitingController.py

import numpy as np
import random
import pygame
import heapq
from controller.Controller import Controller

# ==============================================================================
# 1. A* PLANNER (Giữ nguyên logic cũ, chỉ tinh chỉnh nhỏ)
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
        
        # 8 hướng di chuyển chuẩn
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
                # --- SỬA ĐOẠN NÀY ---
                # Tính khoảng cách Euclidean
                h = np.hypot(nx - self.goal[0], ny - self.goal[1])
                
                # Tie-breaker: Ưu tiên đường thẳng nối từ Start đến Goal
                # Tính tích chéo (cross product) để xem điểm đang xét lệch bao nhiêu so với đường thẳng lý tưởng
                dx1 = cx - self.goal[0]
                dy1 = cy - self.goal[1]
                dx2 = self.start[0] - self.goal[0]
                dy2 = self.start[1] - self.goal[1]
                cross = abs(dx1 * dy2 - dx2 * dy1)
                
                # Cộng thêm phạt nhỏ cho độ lệch (0.001 * cross)
                # Nhân h với 1.001 để A* "tham lam" hơn, hướng về đích nhanh hơn
                heuristic = h * 1.001 + cross * 0.001
                
                # --------------------

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
        check_r = self.robot_radius * 1.1 # Margin an toàn cho static
        for obs in self.obstacles:
            # Chỉ check static obstacles
            r = pygame.Rect(obs.x - obs.width/2 - check_r, obs.y - obs.height/2 - check_r,
                            obs.width + 2*check_r, obs.height + 2*check_r)
            if r.collidepoint(x, y): return False
        return True

# ==============================================================================
# 2. WAITING RULE (Sửa lại logic tính toán thời gian va chạm chuẩn đơn vị)
# ==============================================================================
class WaitingRule:
    def __init__(self, prediction_horizon=15):
        self.prediction_horizon = prediction_horizon

    def get_time_to_collision(self, robot, direction, dynamic_obstacles):
        # Robot position (pixel)
        rob_pos = np.array([robot.x, robot.y])
        
        # Robot velocity: direction (unit vector hoặc grid vector) * cell_size
        # Ở đây giả sử direction là (1,0) hoặc (0.7, 0.7) -> nhân với cell_size ra px/step
        rob_vel = np.array(direction) * robot.cell_size
        
        # Radius an toàn: Robot + Obstacle + Margin
        # Giả sử obstacle trung bình cỡ 20px
        safe_dist = robot.radius + 20 

        min_ttc = float('inf')

        for obs in dynamic_obstacles:
            obs_pos = np.array([obs.x, obs.y])
            obs_vel = np.array(obs.velocity) # Pixel per step

            # Check tương lai
            for t in range(1, self.prediction_horizon + 1):
                f_rob = rob_pos + rob_vel * t
                f_obs = obs_pos + obs_vel * t
                
                dist = np.linalg.norm(f_rob - f_obs)
                if dist < safe_dist:
                    if t < min_ttc: min_ttc = t
                    break # Va chạm với obs này thì không cần check xa hơn cho obs này
        
        return min_ttc if min_ttc != float('inf') else None

# ==============================================================================
# 3. CONTROLLER CHÍNH
# ==============================================================================
class SmartAvoidanceController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=False, model_path=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self.waiting_rule = WaitingRule(prediction_horizon=8)
        self.reset()

    def reset(self):
        self.known_static_obstacles = []
        self.known_ids = set()
        self.current_path = None
        self.target_waypoint_index = 0
        self.stuck_counter = 0
        self.last_position = None
        self.reversing_steps = 0
        self.is_waiting_for_dynamic = False # Cờ đánh dấu đang chủ động chờ
        self.last_chosen_direction = (0, 0)

    def _update_vision(self, robot, all_obstacles):
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

    def make_decision(self, robot, obstacles):
        robot_pos = (robot.x, robot.y)
        dynamic_obstacles = [o for o in obstacles if not o.static]
        
        # 1. Update Vision & Check Path Integrity
        found_new = self._update_vision(robot, obstacles)
        if found_new: 
            self.current_path = None # Replan nếu thấy đường mới

        # 2. Xử lý kẹt cứng (Stuck Handling)
        if self.reversing_steps > 0:
            self.reversing_steps -= 1
            return self._find_escape_direction(robot, obstacles)

        # Kiểm tra nếu robot đứng yên (nhưng KHÔNG PHẢI do đang chủ động chờ vật cản)
        if self.last_position and np.linalg.norm(np.array(robot_pos) - np.array(self.last_position)) < 1.0:
            if not self.is_waiting_for_dynamic:
                self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = robot_pos

        # Nếu kẹt quá lâu -> Replan (Panic mode)
        if self.stuck_counter > 15:
            print("🚨 Stuck! Reversing...")
            self.reversing_steps = 10
            self.current_path = None
            self.stuck_counter = 0
            return self._find_escape_direction(robot, obstacles)

        # 3. Planning (A*)
        if not self.current_path or self.target_waypoint_index >= len(self.current_path):
            planner = AStarPlanner(robot_pos, self.goal, self.known_static_obstacles, 
                                 robot.env_padding*2 + 32*robot.cell_size, 
                                 robot.env_padding*2 + 32*robot.cell_size, robot.radius)
            self.current_path = planner.plan()
            self.target_waypoint_index = 0
            if not self.current_path:
                return (0,0) # Không tìm thấy đường thì đứng yên

        # 4. Path Following & Dynamic Avoidance
        if self.current_path:
            # Chọn waypoint tiếp theo
            target = self.current_path[self.target_waypoint_index]
            if np.linalg.norm(np.array(robot_pos) - np.array(target)) < robot.cell_size:
                if self.target_waypoint_index < len(self.current_path) - 1:
                    self.target_waypoint_index += 1
                    target = self.current_path[self.target_waypoint_index]

            # Vector hướng lý tưởng về đích (A*)
            dx, dy = target[0] - robot_pos[0], target[1] - robot_pos[1]
            ideal_angle = np.arctan2(dy, dx)
            ideal_vector = np.array([np.cos(ideal_angle), np.sin(ideal_angle)])

            # --- QUYẾT ĐỊNH HƯỚNG ĐI DỰA TRÊN GRID (QUAN TRỌNG) ---
            # Thay vì trả về vector float, ta chọn hướng Grid tốt nhất (Discrete Choice)
            best_move = (0, 0)
            best_score = -float('inf')
            self.is_waiting_for_dynamic = False

            # Lọc danh sách các hướng hợp lệ (Grid moves)
            valid_moves = []
            
            # Thêm move đứng yên (Wait)
            valid_moves.append(((0,0), 0)) # move, score penalty

            for d in self.directions: # 8 hướng xung quanh
                # 1. Check Static Obstacles (Tường)
                if not self._is_move_safe_static(robot, d, self.known_static_obstacles):
                    continue
                
                # 2. Check Dynamic Obstacles (TTC)
                ttc = self.waiting_rule.get_time_to_collision(robot, d, dynamic_obstacles)
                
                # Nếu va chạm quá gần (< 3 bước) -> Loại bỏ ngay
                if ttc is not None and ttc < 4:
                    continue 

                # 3. Tính điểm (Score) - SỬA LẠI ĐỂ MƯỢT HƠN
                d_vec = np.array(d)
                d_norm = d_vec / np.linalg.norm(d_vec)
                alignment = np.dot(d_norm, ideal_vector)
                
                score = alignment * 2.0 
                
                # --- THÊM: Quán tính (Inertia) ---
                # Nếu hướng d trùng với hướng vừa đi frame trước, cộng thêm điểm thưởng
                if d == self.last_chosen_direction:
                    score += 0.5 
                
                # Penalty an toàn
                if ttc is not None:
                    score -= 15.0 / (ttc + 0.1) # Phạt nặng hơn nếu có vật cản
                
                valid_moves.append((d, score))

            # Chọn move có điểm cao nhất
            if valid_moves:
                best_move, score = max(valid_moves, key=lambda x: x[1])
                self.last_chosen_direction = best_move 
                
                # Nếu move tốt nhất là đứng yên hoặc điểm quá thấp -> Chuyển trạng thái chờ
                if best_move == (0,0):
                    self.is_waiting_for_dynamic = True
                    print("✋ Waiting for obstacle...")
                elif score < -5: # Tất cả các hướng đều nguy hiểm
                     best_move = (0,0)
                     self.is_waiting_for_dynamic = True
                     print("✋ Too dangerous, forcing wait.")
                
                return best_move
            else:
                # Không còn đường nào thoát (bị vây kín)
                self.is_waiting_for_dynamic = True
                return (0,0)

        return (0,0)

    def _is_move_safe_static(self, robot, direction, obstacles):
        # Kiểm tra va chạm vật lý với tường/vật cản tĩnh
        # Giả lập vị trí tiếp theo
        next_x = robot.x + direction[0] * robot.cell_size
        next_y = robot.y + direction[1] * robot.cell_size
        
        # Check biên
        if not (robot.env_padding < next_x < robot.env_padding + self.grid_width * robot.cell_size and
                robot.env_padding < next_y < robot.env_padding + self.grid_height * robot.cell_size):
            return False
            
        robot_rect = pygame.Rect(next_x - robot.radius, next_y - robot.radius, 
                               robot.radius*2, robot.radius*2)
        
        for obs in obstacles:
            obs_rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
            if robot_rect.colliderect(obs_rect):
                return False
        return True

    def _find_escape_direction(self, robot, obstacles):
        # Thoát kẹt đơn giản bằng cách tìm hướng trống trải nhất
        best_d = (0,0)
        max_dist = -1
        dirs = list(self.directions)
        random.shuffle(dirs)
        
        for d in dirs:
            if not self._is_move_safe_static(robot, d, obstacles): continue
            # Raycast đơn giản
            dist = 0
            for k in range(1, 6):
                nx = robot.x + d[0]*robot.cell_size*k
                ny = robot.y + d[1]*robot.cell_size*k
                # Check đơn giản va chạm điểm
                collision = False
                for obs in obstacles:
                    if obs.static and (obs.x - obs.width/2 < nx < obs.x + obs.width/2 and
                                     obs.y - obs.height/2 < ny < obs.y + obs.height/2):
                        collision = True
                        break
                if collision: break
                dist += 1
            
            if dist > max_dist:
                max_dist = dist
                best_d = d
        return best_d