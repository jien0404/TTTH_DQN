# controller/WaitingController.py

import numpy as np
import random
import pygame
from controller.Controller import Controller

# ==============================================================================
# CÁC LỚP TIỆN ÍCH CHO THUẬT TOÁN (RRT và Waiting Rule)
# ==============================================================================

class RRTNode:
    """Nút trong cây RRT"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRTPlanner:
    """
    Thuật toán lập kế hoạch đường đi RRT đã tối ưu.
    Hỗ trợ: Coordinate correction, Tolerance margin, Path smoothing.
    """
    def __init__(self, start, goal, obstacles, grid_width, grid_height, robot_radius,
             expand_dist=20, max_iter=2000, forbidden_zones=None):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.robot_radius = robot_radius
        self.expand_dist = expand_dist
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.forbidden_zones = forbidden_zones if forbidden_zones else []

    def plan(self):
        """Thực hiện tìm đường RRT"""
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            
            if self._is_node_in_forbidden_zone(rnd_node):
                continue
            
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dist)
            
            if self._is_node_in_forbidden_zone(new_node):
                continue
            
            if self.is_collision_free(new_node, nearest_node):
                self.node_list.append(new_node)
                
                # Nếu gần đích
                if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dist:
                    if self.is_collision_free(new_node, self.goal):
                        self.goal.parent = new_node
                        raw_path = self.generate_final_path()
                        # Tối ưu: Làm mượt đường đi
                        return self.smooth_path(raw_path)
        return None

    def smooth_path(self, path):
        """
        Làm mượt đường đi bằng cách nối tắt các điểm (Pruning).
        Giúp robot đi thẳng trong hành lang thay vì zic-zac.
        """
        if len(path) <= 2:
            return path
            
        smoothed_path = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            last_valid_idx = current_idx + 1
            # Thử nối điểm hiện tại với các điểm xa hơn
            for i in range(len(path) - 1, current_idx + 1, -1):
                # Kiểm tra đường thẳng từ current -> i có va chạm không
                node_a = RRTNode(path[current_idx][0], path[current_idx][1])
                node_b = RRTNode(path[i][0], path[i][1])
                
                if self.is_collision_free(node_b, node_a):
                    last_valid_idx = i
                    break
            
            smoothed_path.append(path[last_valid_idx])
            current_idx = last_valid_idx
            
        return smoothed_path

    def _is_node_in_forbidden_zone(self, node):
        node_pos = np.array([node.x, node.y])
        for zone in self.forbidden_zones:
            dist = np.linalg.norm(node_pos - zone['center'])
            if dist < zone['radius']:
                return True
        return False

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = RRTNode(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.parent = from_node
        if extend_length > d:
            extend_length = d
        new_node.x += extend_length * np.cos(theta)
        new_node.y += extend_length * np.sin(theta)
        return new_node

    def generate_final_path(self):
        path = [[self.goal.x, self.goal.y]]
        node = self.goal
        while node.parent is not None:
            path.append([node.parent.x, node.parent.y])
            node = node.parent
        return path[::-1]

    def get_random_node(self):
        if random.randint(0, 100) > 10: # Tăng Goal Bias lên 10% để hội tụ nhanh hơn
            rnd = RRTNode(random.uniform(0, self.grid_width),
                          random.uniform(0, self.grid_height))
        else:
            rnd = RRTNode(self.goal.x, self.goal.y)
        return rnd

    def is_collision_free(self, new_node, nearest_node):
        """
        Kiểm tra va chạm với vật cản được 'thổi phồng' (Inflated).
        Đã sửa lỗi tọa độ và thêm dung sai để lọt khe hẹp.
        """
        # QUAN TRỌNG: Sử dụng bán kính nhỏ hơn thực tế một chút (Tolerance)
        # để cho phép robot đi qua các khe "vừa khít" mà không bị chặn bởi sai số dấu phẩy động.
        check_radius = self.robot_radius * 0.95 
        
        p1 = np.array([nearest_node.x, nearest_node.y])
        p2 = np.array([new_node.x, new_node.y])

        for obs in self.obstacles:
            if obs.static:
                # SỬA LỖI TỌA ĐỘ: obs.x là tâm -> chuyển về topleft
                obs_left = obs.x - obs.width / 2
                obs_top = obs.y - obs.height / 2
                
                # Tạo Rect đã mở rộng (Minkowski Sum xấp xỉ)
                inflated_rect = pygame.Rect(
                    obs_left - check_radius,
                    obs_top - check_radius,
                    obs.width + 2 * check_radius,
                    obs.height + 2 * check_radius
                )
                
                # Kiểm tra đoạn thẳng có cắt Rect không
                if inflated_rect.clipline(p1, p2):
                    return False
                    
        return True

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.goal.x, y - self.goal.y])
    
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

class WaitingRule:
    """Quy tắc chờ nâng cao."""
    def __init__(self, prediction_horizon=10, safety_margin=30):
        self.prediction_horizon = prediction_horizon
        self.safety_margin = safety_margin

    def get_time_to_collision(self, robot, intended_direction, dynamic_obstacles):
        robot_pos = np.array([robot.x, robot.y])
        min_time_to_collision = float('inf')
        collision_found = False

        cell_size = robot.cell_size
        assumed_speed = cell_size * 0.8
        intended_velocity = np.array(intended_direction) * assumed_speed

        for obs in dynamic_obstacles:
            obs_pos = np.array([obs.x, obs.y])
            obs_vel = np.array(obs.velocity)

            for t in range(1, self.prediction_horizon + 1):
                future_robot_pos = robot_pos + intended_velocity * t
                future_obs_pos = obs_pos + obs_vel * t

                dist = np.linalg.norm(future_robot_pos - future_obs_pos)
                
                # Điều chỉnh safety margin dựa trên kích thước thực
                # Giả sử obs cũng có bán kính tương đương robot
                combined_radius = robot.radius * 2 + 5 # +5 buffer
                
                if dist < combined_radius:
                    vec_to_robot = robot_pos - obs_pos
                    if np.dot(obs_vel, vec_to_robot) > 0:
                        min_time_to_collision = min(min_time_to_collision, t)
                        collision_found = True
                        break 
        
        return min_time_to_collision if collision_found else None

# ==============================================================================
# CONTROLLER CHÍNH
# ==============================================================================

class WaitingController(Controller):
    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height, is_training=False, model_path=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("Initializing Optimized Waiting & RRT Controller...")
        self.waiting_rule = WaitingRule(
            prediction_horizon=6,
            safety_margin=self.cell_size * 1.5
        )
        
        self.current_path = None
        self.target_waypoint_index = 0
        self.is_waiting = False
        self.replanning_cooldown = 0
        self.known_static_obstacles = []
        self.stuck_counter = 0
        self.last_position = None
        
        self.forbidden_zones = []
        self.position_history = []
        self.replan_count = 0
        
        # --- MỚI: Biến đếm lùi xe ---
        self.reversing_steps = 0  # Số bước robot buộc phải đi lùi
        self.last_valid_direction = (0, 0) # Hướng đi hợp lệ cuối cùng

    def _find_safe_retreat_direction(self, robot, all_obstacles):
        """
        Tìm hướng thoát hiểm thông minh:
        1. Ưu tiên hướng có không gian rộng nhất (thoát khỏi khe hẹp).
        2. Ưu tiên hướng ngược lại với hướng đang bị kẹt.
        """
        robot_pos = np.array([robot.x, robot.y])
        static_obstacles = [obs for obs in all_obstacles if obs.static]
        
        best_direction = None
        max_clearance = -1

        # Quét tất cả 8 hướng xung quanh
        # Xáo trộn nhẹ để nếu các hướng ngang nhau thì không bị bias
        candidates = list(self.directions)
        random.shuffle(candidates)

        for direction in candidates:
            # 1. Kiểm tra cơ bản: Có va chạm ngay lập tức không?
            if not self._is_move_safe(robot, direction, static_obstacles):
                continue

            # 2. Tính điểm "thoáng" (Clearance Score)
            # Kiểm tra xem hướng này có thể đi xa bao nhiêu bước mà không đụng tường
            # Hoặc kiểm tra xem tại hướng đó, bán kính an toàn có lớn hơn không
            score = 0
            
            # Check 3 bước tiếp theo theo hướng đó
            for step in range(1, 4):
                future_pos = (
                    robot.x + direction[0] * self.cell_size * step,
                    robot.y + direction[1] * self.cell_size * step
                )
                
                # Tạo một hình chữ nhật kiểm tra an toàn tại vị trí tương lai
                # Nếu đặt được hình chữ nhật ở đó -> cộng điểm
                # Mẹo: Kiểm tra va chạm với bán kính LỚN HƠN robot một chút để ưu tiên chỗ rộng
                step_safe = True
                
                # Check biên
                if not (self.env_padding < future_pos[0] < self.env_padding + self.grid_width * self.cell_size and
                        self.env_padding < future_pos[1] < self.env_padding + self.grid_height * self.cell_size):
                    step_safe = False
                
                # Check vật cản
                if step_safe:
                    for obs in static_obstacles:
                        obs_left = obs.x - obs.width / 2
                        obs_top = obs.y - obs.height / 2
                        obs_rect = pygame.Rect(obs_left, obs_top, obs.width, obs.height)
                        if self._circle_collides_rect(future_pos, robot.radius * 1.2, obs_rect):
                            step_safe = False
                            break
                
                if step_safe:
                    score += 1
                else:
                    break # Gặp vật cản thì dừng tính điểm hướng này
            
            # Chọn hướng có điểm cao nhất (đi được xa nhất/rộng nhất)
            if score > max_clearance:
                max_clearance = score
                best_direction = direction
        
        return best_direction

    def make_decision(self, robot, obstacles):
        self.replanning_cooldown = max(0, self.replanning_cooldown - 1)
        robot_pos = (robot.x, robot.y)

        static_obstacles = [obs for obs in obstacles if obs.static]
        dynamic_obstacles = [obs for obs in obstacles if not obs.static]
        
        if not self.known_static_obstacles:
            self.known_static_obstacles = static_obstacles

        # --- ƯU TIÊN 1: CHẾ ĐỘ LÙI XE (ESCAPE MODE) ---
        # Nếu đang trong chuỗi hành động lùi, thực hiện ngay không cần suy nghĩ
        if self.reversing_steps > 0:
            print(f"🔙 Reversing to escape trap... ({self.reversing_steps} left)")
            retreat_dir = self._find_safe_retreat_direction(robot, obstacles)
            self.reversing_steps -= 1
            if retreat_dir:
                return retreat_dir
            else:
                # Nếu lùi cũng tắc -> Hết cách, đứng yên chờ
                return (0, 0)

        # --- 2. Kiểm tra Stuck / Loop ---
        if self.last_position is not None:
            if np.linalg.norm(np.array(robot_pos) - np.array(self.last_position)) < 2.0:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        self.last_position = robot_pos
        is_looping = self._detect_position_loop(robot_pos)

        # Xử lý khi bị kẹt -> KÍCH HOẠT LÙI XE
        STUCK_THRESHOLD = 10 # Giảm ngưỡng xuống để phản ứng nhanh hơn
        if (self.stuck_counter >= STUCK_THRESHOLD or is_looping) and self.replanning_cooldown == 0:
            print(f"🚨 STUCK DETECTED! Initiating reverse maneuver.")
            
            # 1. Đánh dấu vùng hiện tại là vùng cấm (tâm hơi lệch về phía trước mặt robot để chặn đường vào)
            if self.current_path and self.target_waypoint_index < len(self.current_path):
                 # Chặn đường phía trước mặt robot
                block_pos = self.current_path[self.target_waypoint_index]
            else:
                block_pos = robot_pos
                
            self._add_forbidden_zone(block_pos, radius=self.cell_size * 2.5)
            
            # 2. Xóa đường cũ
            self.current_path = None
            self.stuck_counter = 0
            self.position_history.clear()
            
            # 3. Kích hoạt chế độ lùi trong 15 frames
            self.reversing_steps = 15 
            
            # Trả về luôn hướng lùi cho frame này
            return self._find_safe_retreat_direction(robot, obstacles) or (0,0)

        # --- 3. Lập kế hoạch đường đi (RRT) ---
        path_needed = self.current_path is None or self.target_waypoint_index >= len(self.current_path)
        
        if path_needed and self.replanning_cooldown == 0:
            self.replan_count += 1
            if self.replan_count % 5 == 0:
                self.forbidden_zones.clear()

            # Tinh chỉnh tham số RRT cho trường hợp thoát ngõ cụt
            # Giảm expand_dist để len lỏi tốt hơn trong khe hẹp
            planner = RRTPlanner(
                start=robot_pos,
                goal=self.goal,
                obstacles=self.known_static_obstacles,
                grid_width=robot.env_padding * 2 + 32 * self.cell_size,
                grid_height=robot.env_padding * 2 + 32 * self.cell_size,
                robot_radius=robot.radius,
                expand_dist=15, # Giảm xuống để step nhỏ hơn
                max_iter=3000,  # Tăng iter để tìm đường vòng xa hơn
                forbidden_zones=self.forbidden_zones
            )
            
            self.current_path = planner.plan()
            self.target_waypoint_index = 0
            self.replanning_cooldown = 15
            
            if self.current_path is None:
                # Nếu RRT vẫn không tìm ra đường, có thể ta vẫn đang quá sâu trong bẫy
                # Lùi thêm một chút nữa
                print("❌ RRT Failed. Reversing more.")
                self.reversing_steps = 5
                return self._find_safe_retreat_direction(robot, obstacles) or (0,0)

        # --- 3. Bám theo đường dẫn ---
        target_waypoint = self.current_path[self.target_waypoint_index]
        dist_to_waypoint = np.linalg.norm(np.array(robot_pos) - np.array(target_waypoint))
        
        if dist_to_waypoint < self.cell_size and self.target_waypoint_index < len(self.current_path) - 1:
            self.target_waypoint_index += 1
            target_waypoint = self.current_path[self.target_waypoint_index]

        dx = target_waypoint[0] - robot_pos[0]
        dy = target_waypoint[1] - robot_pos[1]
        target_angle = np.arctan2(dy, dx)
        
        valid_directions = []
        for direction in self.directions:
            if self._is_move_safe(robot, direction, static_obstacles):
                dir_angle = np.arctan2(direction[1], direction[0])
                angle_diff = abs(target_angle - dir_angle)
                if angle_diff > np.pi: angle_diff = 2 * np.pi - angle_diff
                valid_directions.append((direction, angle_diff))
        
        if not valid_directions:
            # Nếu bị kẹt cục bộ -> Kích hoạt lùi ngay lập tức
            self.reversing_steps = 5
            return self._find_safe_retreat_direction(robot, obstacles) or (0,0)
        
        best_direction = min(valid_directions, key=lambda item: item[1])[0]

        intended_unit_vec = (best_direction[0], best_direction[1])
        time_to_collision = self.waiting_rule.get_time_to_collision(robot, intended_unit_vec, dynamic_obstacles)
        
        if time_to_collision is not None and time_to_collision <= 3:
            # Logic tránh vật cản động giữ nguyên
            valid_directions.sort(key=lambda x: x[1])
            for direction, _ in valid_directions:
                unit_vec = (direction[0], direction[1])
                ttc = self.waiting_rule.get_time_to_collision(robot, unit_vec, dynamic_obstacles)
                if ttc is None or ttc > 5:
                    return direction
            return (0, 0)
        
        return best_direction

    def _circle_collides_rect(self, circle_center, circle_radius, rect):
        """Kiểm tra va chạm hình tròn - hình chữ nhật chính xác."""
        closest_x = max(rect.left, min(circle_center[0], rect.right))
        closest_y = max(rect.top, min(circle_center[1], rect.bottom))
        
        distance_x = circle_center[0] - closest_x
        distance_y = circle_center[1] - closest_y
        
        return (distance_x ** 2 + distance_y ** 2) < (circle_radius ** 2)

    def _is_move_safe(self, robot, direction, static_obstacles):
        """
        Kiểm tra va chạm local cho bước đi tiếp theo.
        ĐÃ SỬA: Chuyển đổi tọa độ tâm -> topleft cho Rect.
        """
        next_center_x = robot.x + direction[0] * self.cell_size
        next_center_y = robot.y + direction[1] * self.cell_size
        next_center_pos = (next_center_x, next_center_y)

        # 1. Kiểm tra biên
        if not (self.env_padding + robot.radius <= next_center_x <= self.env_padding + self.grid_width * self.cell_size - robot.radius and
                self.env_padding + robot.radius <= next_center_y <= self.env_padding + self.grid_height * self.cell_size - robot.radius):
            return False

        # 2. Kiểm tra va chạm vật cản
        # Sử dụng hệ số an toàn nhỏ hơn 1 chút (0.9) ở local planner 
        # để cho phép robot áp sát tường khi cần thiết
        safe_radius = robot.radius * 0.95

        for obs in static_obstacles:
            # SỬA LỖI: obs.x là tâm, Rect cần topleft
            obs_left = obs.x - obs.width / 2
            obs_top = obs.y - obs.height / 2
            
            obstacle_rect = pygame.Rect(obs_left, obs_top, obs.width, obs.height)
            
            if self._circle_collides_rect(next_center_pos, safe_radius, obstacle_rect):
                return False
        
        return True
    
    def _detect_position_loop(self, robot_pos, window_size=40, unique_threshold=6):
        """Phát hiện robot đi vòng tròn."""
        self.position_history.append(tuple(robot_pos))
        if len(self.position_history) > window_size:
            self.position_history.pop(0)
        
        if len(self.position_history) < window_size:
            return False
        
        # Làm tròn vị trí để gom nhóm các điểm gần nhau
        rounded_history = [(int(p[0]), int(p[1])) for p in self.position_history]
        unique_positions = len(set(rounded_history))
        
        return unique_positions < unique_threshold

    def _add_forbidden_zone(self, center_pos, radius=None):
        if radius is None:
            radius = self.cell_size * 2
        
        self.forbidden_zones.append({
            'center': np.array(center_pos),
            'radius': radius
        })
        if len(self.forbidden_zones) > 3: # Chỉ giữ 3 vùng gần nhất
            self.forbidden_zones.pop(0)