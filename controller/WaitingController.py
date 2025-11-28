# controller/WaitingController.py

import numpy as np
import random
import pygame
import heapq
from controller.Controller import Controller

# ==============================================================================
# 1. A* PLANNER 
# ==============================================================================

class AStarPlanner:
    def __init__(self, start, goal, obstacles, grid_width, grid_height, robot_radius):
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.width = grid_width
        self.height = grid_height
        # Giảm độ phân giải xuống một chút để tìm đường chi tiết hơn
        self.resolution = max(1, int(robot_radius * 0.5)) 

    def plan(self):
        open_set = []
        # PriorityQueue: (f_score, g_score, x, y, path)
        heapq.heappush(open_set, (0, 0, self.start[0], self.start[1], [self.start]))
        visited = {} 

        # 8 hướng di chuyển
        motions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        while open_set:
            f, g, cx, cy, path = heapq.heappop(open_set)

            # Check đến đích
            if np.hypot(cx - self.goal[0], cy - self.goal[1]) <= self.resolution * 1.5:
                path.append(self.goal)
                # --- CẢI TIẾN: Làm mượt đường đi trước khi trả về ---
                return self.smooth_path(path)

            # Grid quantization
            grid_pos = (int(cx // self.resolution), int(cy // self.resolution))
            if grid_pos in visited and visited[grid_pos] <= g:
                continue
            visited[grid_pos] = g

            for dx, dy in motions:
                nx = cx + dx * self.resolution
                ny = cy + dy * self.resolution
                
                if not (0 <= nx <= self.width and 0 <= ny <= self.height):
                    continue

                if not self._is_safe(nx, ny):
                    continue

                move_cost = np.hypot(dx, dy)
                new_g = g + move_cost
                heuristic = np.hypot(nx - self.goal[0], ny - self.goal[1])
                
                new_path = list(path)
                new_path.append((nx, ny))
                heapq.heappush(open_set, (new_g + heuristic, new_g, nx, ny, new_path))
        
        return None

    def smooth_path(self, path):
        """
        Kỹ thuật Greedy Line-of-Sight Smoothing.
        Nối tắt các điểm nếu đường thẳng giữa chúng an toàn.
        """
        if len(path) <= 2:
            return path
            
        smoothed_path = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Tìm điểm xa nhất có thể kết nối thẳng mà không va chạm
            last_valid_idx = current_idx + 1
            
            # Kiểm tra ngược từ cuối về hiện tại để tìm điểm xa nhất (Greedy)
            # Giới hạn tầm nhìn (lookahead) khoảng 10-15 điểm để tối ưu hiệu năng
            check_range = min(len(path), current_idx + 20) 
            
            for i in range(check_range - 1, current_idx, -1):
                target_point = path[i]
                if self._is_line_safe(path[current_idx], target_point):
                    last_valid_idx = i
                    break
            
            smoothed_path.append(path[last_valid_idx])
            current_idx = last_valid_idx
            
        return smoothed_path

    def _is_line_safe(self, start, end):
        x1, y1 = start
        x2, y2 = end
        dist = np.hypot(x2 - x1, y2 - y1)
        if dist == 0: return True
        
        # Tăng độ phân giải kiểm tra lên để không bỏ sót điểm va chạm trong khe hẹp
        steps = int(dist / (self.robot_radius * 0.5)) + 1 
        
        for i in range(steps + 1):
            t = i / steps
            check_x = x1 + (x2 - x1) * t
            check_y = y1 + (y2 - y1) * t
            
            if not self._is_safe(check_x, check_y):
                return False
        return True

    def _is_safe(self, x, y):
        # Margin an toàn cho Global Planner (A*)
        # Cần lớn hơn một chút để khi làm mượt không bị sát tường quá
        check_radius = self.robot_radius * 1.1
        
        for obs in self.obstacles:
            # Tính toán bao va chạm
            obs_left = obs.x - obs.width / 2
            obs_top = obs.y - obs.height / 2
            
            inflated_rect = pygame.Rect(
                obs_left - check_radius,
                obs_top - check_radius,
                obs.width + 2 * check_radius,
                obs.height + 2 * check_radius
            )
            if inflated_rect.collidepoint(x, y):
                return False
        return True

# ==============================================================================
# 2. WAITING RULE
# ==============================================================================

class WaitingRule:
    def __init__(self, prediction_horizon=10, safety_margin=30):
        self.prediction_horizon = prediction_horizon
        self.safety_margin = safety_margin

    def get_time_to_collision(self, robot, direction, dynamic_obstacles):
        robot_pos = np.array([robot.x, robot.y])
        # Lấy tốc độ thực tế robot sẽ đi (đã chuẩn hóa là 1 cell/step)
        vel = np.array(direction) * robot.cell_size 
        
        # Bán kính va chạm an toàn (Robot + Obstacle + Margin)
        # Tăng margin lên một chút để robot sợ vật cản hơn
        collision_threshold = robot.radius + 12 

        for obs in dynamic_obstacles:
            obs_pos = np.array([obs.x, obs.y])
            obs_vel = np.array(obs.velocity)

            # KIỂM TRA NGAY BƯỚC ĐẦU TIÊN (Immediate Check)
            # Nếu bước đi tiếp theo gây va chạm ngay lập tức -> TTC = 0
            next_rob = robot_pos + vel
            next_obs = obs_pos + obs_vel
            if np.linalg.norm(next_rob - next_obs) < collision_threshold:
                return 0.5 # Rất nguy hiểm

            for t in range(1, self.prediction_horizon + 1):
                f_rob = robot_pos + vel * t
                f_obs = obs_pos + obs_vel * t
                
                if np.linalg.norm(f_rob - f_obs) < collision_threshold:
                    return t
                    
        return None

# ==============================================================================
# 3. CONTROLLER CHÍNH
# ==============================================================================

class WaitingController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=False, model_path=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("Initializing Fog-of-War Controller (Smoothed A*)...")
        self.waiting_rule = WaitingRule(prediction_horizon=4, safety_margin=self.cell_size * 0.5)
        self.reset()

    def reset(self):
        print("🔄 Controller State Reset")
        self.known_static_obstacles = [] 
        self.known_ids = set()           
        self.current_path = None
        self.target_waypoint_index = 0
        self.replanning_cooldown = 0
        self.stuck_counter = 0
        self.last_position = None
        self.reversing_steps = 0

    def _update_vision(self, robot, all_obstacles):
        found_new = False
        vision_sq = robot.vision ** 2
        
        for obs in all_obstacles:
            if not obs.static: continue 
            if id(obs) in self.known_ids: continue 
            
            dist_sq = (obs.x - robot.x)**2 + (obs.y - robot.y)**2
            if dist_sq <= vision_sq:
                self.known_static_obstacles.append(obs)
                self.known_ids.add(id(obs))
                found_new = True
        return found_new

    def _is_current_path_unsafe(self, robot_pos):
        if not self.current_path: return True
        
        # 1. KIỂM TRA QUAN TRỌNG NHẤT: Đoạn từ Robot đến điểm mốc tiếp theo
        # Nếu ngay trước mặt bị chặn thì phải tìm đường mới ngay
        target_pt = self.current_path[self.target_waypoint_index]
        
        # Dùng planner ảo để check va chạm
        dummy_planner = AStarPlanner((0,0), (0,0), self.known_static_obstacles, 0, 0, 7)
        
        if not dummy_planner._is_line_safe(robot_pos, target_pt):
             print("🚫 Immediate path blocked! (Robot -> Target)")
             return True

        # 2. Kiểm tra các đoạn đường tiếp theo trong tương lai
        start = self.target_waypoint_index
        end = min(len(self.current_path), start + 10)
        
        for i in range(start, end - 1):
            p1 = self.current_path[i]
            p2 = self.current_path[i+1]
            if not dummy_planner._is_line_safe(p1, p2):
                print(f"🚫 Future path segment {i} blocked!")
                return True 
                
        return False
    
    def _apply_dynamic_steering(self, robot, intended_dir, dynamic_obstacles):
        """
        Điều chỉnh hướng đi dựa trên vận tốc của vật cản động (Local Steering).
        Nguyên tắc: Né về phía ngược lại với hướng di chuyển của vật cản.
        """
        robot_pos = np.array([robot.x, robot.y])
        best_dir = np.array(intended_dir)
        
        # Tìm vật cản nguy hiểm nhất
        most_dangerous_obs = None
        min_ttc = float('inf')
        
        # Chỉ xét vật cản trong tầm nhìn và đang có nguy cơ va chạm
        for obs in dynamic_obstacles:
            dist = np.linalg.norm(np.array([obs.x, obs.y]) - robot_pos)
            if dist > robot.vision * 1.2: continue # Quá xa thì kệ
            
            # Dự đoán va chạm
            ttc = self.waiting_rule.get_time_to_collision(robot, tuple(best_dir), [obs])
            
            # Nếu có nguy cơ va chạm gần (dưới 15 bước - khoảng 1.5s)
            if ttc is not None and ttc < 15:
                if ttc < min_ttc:
                    min_ttc = ttc
                    most_dangerous_obs = obs
        
        # Nếu không có mối đe dọa nào quá gần, giữ nguyên hướng
        if most_dangerous_obs is None:
            return intended_dir

        print(f"⚠️ Steering to avoid dynamic obstacle (TTC: {min_ttc})")
        
        # --- TÍNH TOÁN HƯỚNG LÁCH ---
        obs_vel = np.array(most_dangerous_obs.velocity)
        
        # 1. Tính vector pháp tuyến của hướng đi robot (Trái và Phải)
        # Hướng đi: (dx, dy) -> Vuông góc phải: (-dy, dx), Vuông góc trái: (dy, -dx)
        right_normal = np.array([-best_dir[1], best_dir[0]])
        left_normal = np.array([best_dir[1], -best_dir[0]])
        
        # 2. Xem vật cản đang trôi về bên nào so với đường đi của robot
        # Dùng tích vô hướng (Dot Product) để chiếu vận tốc vật cản lên pháp tuyến phải
        # Nếu > 0: Vật cản đang đi sang phải -> Robot nên né sang Trái
        drift_score = np.dot(obs_vel, right_normal)
        
        avoidance_force = np.array([0.0, 0.0])
        strength = 1.5 # Cường độ lách (càng lớn lách càng gắt)
        
        # Logic người dùng yêu cầu:
        if drift_score > 0.1: 
            # Vật cản đi sang phải -> Robot lách sang Trái (đi vòng ra sau lưng hoặc tạt đầu xa)
            avoidance_force = left_normal * strength
        elif drift_score < -0.1:
            # Vật cản đi sang trái -> Robot lách sang Phải
            avoidance_force = right_normal * strength
        else:
            # Vật cản đi trực diện hoặc đứng yên trên đường
            # Lách sang bên nào thoáng hơn (dựa vào vị trí vật cản)
            vec_to_obs = np.array([most_dangerous_obs.x, most_dangerous_obs.y]) - robot_pos
            if np.dot(vec_to_obs, right_normal) > 0:
                avoidance_force = left_normal * strength # Vật cản ở bên phải thì lách trái
            else:
                avoidance_force = right_normal * strength # Vật cản ở bên trái thì lách phải
                
        # 3. Cộng lực lách vào hướng đi chính
        new_dir = best_dir + avoidance_force
        
        # Chuẩn hóa lại vector
        norm = np.linalg.norm(new_dir)
        if norm > 0:
            new_dir = new_dir / norm
            
        return tuple(new_dir)

    def make_decision(self, robot, obstacles):
        self.replanning_cooldown = max(0, self.replanning_cooldown - 1)
        robot_pos = (robot.x, robot.y)
        dynamic_obstacles = [obs for obs in obstacles if not obs.static]

        # 1. Vision
        found_new_obstacle = self._update_vision(robot, obstacles)
        if self.current_path:
            # Truyền robot_pos vào để check đoạn ngay trước mặt
            if found_new_obstacle or self.stuck_counter > 2 or self._is_current_path_unsafe(robot_pos):
                if found_new_obstacle: print("👀 New obstacle found.")
                elif self.stuck_counter > 2: print("⚠️ Robot moving slowly/stuck.")
                else: print("⚠️ Path became unsafe.")
                
                self.current_path = None # Hủy đường cũ

        # 2. Stuck Handling
        if self.reversing_steps > 0:
            self.reversing_steps -= 1
            return self._find_escape_direction(robot, obstacles) or (0,0)

        if self.last_position and np.linalg.norm(np.array(robot_pos) - np.array(self.last_position)) < 1.0:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = robot_pos

        if self.stuck_counter > 10:
            print("🚨 Stuck! Adding virtual obstacle and Reversing.")
            
            # THÊM MỚI: Tạo vật cản ảo tại vị trí đang kẹt (hoặc phía trước mặt)
            # Để lần sau A* biết đường mà né chỗ này ra
            virtual_obs = pygame.Rect(robot.x - 10, robot.y - 10, 20, 20)
            # Lưu ý: Bạn cần tạo một class Obstacle giả hoặc struct tương tự để A* hiểu
            # Ở đây tôi dùng object đơn giản có thuộc tính x, y, width, height, static
            class VirtualObs:
                def __init__(self, x, y):
                    self.x, self.y = x, y
                    self.width, self.height = 20, 20
                    self.static = True
            
            self.known_static_obstacles.append(VirtualObs(robot.x, robot.y))
            
            self.reversing_steps = 15 # Lùi xa hơn chút
            self.current_path = None
            self.stuck_counter = 0
            return self._find_escape_direction(robot, obstacles) or (0,0)

        # 3. Planning
        if (self.current_path is None or self.target_waypoint_index >= len(self.current_path)) and self.replanning_cooldown == 0:
            planner = AStarPlanner(
                start=robot_pos,
                goal=self.goal,
                obstacles=self.known_static_obstacles,
                grid_width=robot.env_padding * 2 + 32 * self.cell_size,
                grid_height=robot.env_padding * 2 + 32 * self.cell_size,
                robot_radius=robot.radius
            )
            self.current_path = planner.plan()
            self.target_waypoint_index = 0
            self.replanning_cooldown = 5 
            
            if not self.current_path:
                self.reversing_steps = 5
                return self._find_escape_direction(robot, obstacles) or (0,0)

        # 4. Path Following (Cải tiến một chút: Pure Pursuit đơn giản)
        if self.current_path:
            # Tìm target point xa hơn một chút để đi mượt hơn
            target_idx = self.target_waypoint_index
            target = self.current_path[target_idx]
            
            # Logic: Nếu đã đến gần waypoint hiện tại, nhắm tới waypoint tiếp theo
            dist_to_current = np.linalg.norm(np.array(robot_pos) - np.array(target))
            if dist_to_current < self.cell_size:
                if self.target_waypoint_index < len(self.current_path) - 1:
                    self.target_waypoint_index += 1
                    target = self.current_path[self.target_waypoint_index]
            
            # Nếu đường đi đã được smooth, các điểm cách xa nhau
            # Ta cứ nhắm thẳng vào target waypoint
            dx, dy = target[0] - robot_pos[0], target[1] - robot_pos[1]
            target_angle = np.arctan2(dy, dx)
            
            valid_dirs = []
            for d in self.directions:
                if self._is_move_safe(robot, d, obstacles): 
                    da = np.arctan2(d[1], d[0])
                    diff = abs(target_angle - da)
                    if diff > np.pi: diff = 2*np.pi - diff
                    valid_dirs.append((d, diff))
            
            if valid_dirs:
                best_dir_static = min(valid_dirs, key=lambda x: x[1])[0]
                
                # --- SỬA ĐỔI: ÁP DỤNG STEERING BEHAVIOR ---
                
                # 1. Tính hướng né vật cản động
                steered_dir = self._apply_dynamic_steering(robot, best_dir_static, dynamic_obstacles)
                
                # 2. Kiểm tra xem hướng đã né này có an toàn với TƯỜNG (vật cản tĩnh) không?
                # Robot lách vật cản động nhưng không được đâm vào tường
                if self._is_move_safe(robot, steered_dir, self.known_static_obstacles):
                    # Nếu an toàn, đi theo hướng đã lách
                    # Kiểm tra lại lần cuối xem hướng lách này có va chạm ngay lập tức với dynamic obs không (trường hợp quá gần)
                    ttc = self.waiting_rule.get_time_to_collision(robot, steered_dir, dynamic_obstacles)
                    if ttc is None or ttc > 2: # > 2 bước là đủ an toàn để lướt qua
                         return steered_dir
                
                # 3. Nếu hướng lách bị chặn bởi tường, hoặc vẫn sẽ đâm vào vật cản động
                # Thì đành phải phanh lại (Wait) hoặc dùng A* detour (như code trước)
                print("⚠️ Steering blocked or unsafe. Waiting/Braking.")
                return (0,0)
        
        return (0,0)

    def _find_escape_direction(self, robot, obstacles):
        robot_pos = np.array([robot.x, robot.y])
        static_obs = [o for o in obstacles if o.static]
        best_d = None
        max_dist = -1
        candidates = list(self.directions)
        random.shuffle(candidates)
        
        for d in candidates:
            if not self._is_move_safe(robot, d, static_obs): continue
            dist = 0
            for i in range(1, 5):
                check_pos = robot_pos + np.array(d) * self.cell_size * i
                collision = False
                for obs in static_obs:
                     # Check nhanh
                     if (obs.x - obs.width/2 - 5 < check_pos[0] < obs.x + obs.width/2 + 5 and
                         obs.y - obs.height/2 - 5 < check_pos[1] < obs.y + obs.height/2 + 5):
                         collision = True
                         break
                if collision: break
                dist += 1
            if dist > max_dist:
                max_dist = dist
                best_d = d
        return best_d

    def _circle_collides_rect(self, circle_center, circle_radius, rect):
        closest_x = max(rect.left, min(circle_center[0], rect.right))
        closest_y = max(rect.top, min(circle_center[1], rect.bottom))
        return ((circle_center[0] - closest_x)**2 + (circle_center[1] - closest_y)**2) < (circle_radius**2)

    def _is_move_safe(self, robot, direction, obstacles):
        next_pos = (robot.x + direction[0] * self.cell_size, 
                   robot.y + direction[1] * self.cell_size)
        if not (self.env_padding < next_pos[0] < self.env_padding + self.grid_width * self.cell_size and
                self.env_padding < next_pos[1] < self.env_padding + self.grid_height * self.cell_size):
            return False
        safe_r = robot.radius * 0.95
        for obs in obstacles:
            if not obs.static: continue
            rect = pygame.Rect(obs.x - obs.width/2, obs.y - obs.height/2, obs.width, obs.height)
            if self._circle_collides_rect(next_pos, safe_r, rect):
                return False
        return True