# controller/IndoorAdaptedController.py

import numpy as np
import pygame
import random
from controller.Controller import Controller


# ==============================================================================
# RRT PLANNER - TỐI ƯU HÓA (GIỮ NGUYÊN)
# ==============================================================================

class RRTNode:
    """Nút trong cây RRT"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


class OptimizedRRTPlanner:
    """
    RRT được tối ưu với:
    - Goal bias cao
    - Path smoothing tốt
    - Rewiring (giống RRT*)
    """

    def __init__(self, start, goal, obstacles, grid_width, grid_height, robot_radius,
                 expand_dist=20, max_iter=5000, goal_sample_rate=20):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.robot_radius = robot_radius
        self.expand_dist = expand_dist
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.node_list = [self.start]

    def plan(self):
        """Thực hiện tìm đường RRT với rewiring"""
        for i in range(self.max_iter):
            # Sample random node (với goal bias cao)
            rnd_node = self.get_random_node()

            # Tìm nearest node
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # Steer về phía random node
            new_node = self.steer(nearest_node, rnd_node, self.expand_dist)

            # Kiểm tra collision
            if self.is_collision_free(new_node, nearest_node):
                # Rewiring: Tìm parent tốt hơn trong vùng lân cận
                near_indices = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indices)

                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indices)

                    # Kiểm tra đã đến goal chưa
                    if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dist:
                        if self.is_collision_free(new_node, self.goal):
                            self.goal.parent = new_node
                            raw_path = self.generate_final_path()
                            # Smooth path nhiều lần
                            smoothed = self.smooth_path(raw_path)
                            smoothed = self.smooth_path(smoothed)  # Smooth 2 lần
                            return smoothed

        return None

    def choose_parent(self, new_node, near_indices):
        """Chọn parent có cost nhỏ nhất"""
        if not near_indices:
            return new_node

        costs = []
        for i in near_indices:
            near_node = self.node_list[i]
            if self.is_collision_free(new_node, near_node):
                cost = self.calc_cost(near_node) + self.calc_distance(near_node, new_node)
                costs.append((cost, near_node))

        if not costs:
            return new_node

        min_cost, best_parent = min(costs, key=lambda x: x[0])
        new_node.parent = best_parent
        return new_node

    def rewire(self, new_node, near_indices):
        """Rewire các node gần để giảm cost"""
        for i in near_indices:
            near_node = self.node_list[i]

            if self.is_collision_free(new_node, near_node):
                new_cost = self.calc_cost(new_node) + self.calc_distance(new_node, near_node)
                old_cost = self.calc_cost(near_node)

                if new_cost < old_cost:
                    near_node.parent = new_node

    def calc_cost(self, node):
        """Tính cost từ start đến node"""
        cost = 0
        current = node
        while current.parent:
            cost += self.calc_distance(current, current.parent)
            current = current.parent
        return cost

    def calc_distance(self, node1, node2):
        """Tính khoảng cách Euclidean"""
        return np.hypot(node1.x - node2.x, node1.y - node2.y)

    def find_near_nodes(self, new_node):
        """Tìm các node trong bán kính"""
        r = self.expand_dist * 3  # Bán kính tìm kiếm
        return [i for i, node in enumerate(self.node_list)
                if self.calc_distance(node, new_node) <= r]

    def smooth_path(self, path):
        """Làm mượt đường đi bằng cách nối tắt"""
        if len(path) <= 2:
            return path

        smoothed_path = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            last_valid_idx = current_idx + 1

            # Thử nối xa nhất có thể
            for i in range(len(path) - 1, current_idx + 1, -1):
                node_a = RRTNode(path[current_idx][0], path[current_idx][1])
                node_b = RRTNode(path[i][0], path[i][1])

                if self.is_collision_free(node_b, node_a):
                    last_valid_idx = i
                    break

            smoothed_path.append(path[last_valid_idx])
            current_idx = last_valid_idx

        return smoothed_path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Mở rộng từ from_node về phía to_node"""
        new_node = RRTNode(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(from_node, to_node)
        new_node.parent = from_node

        if extend_length > d:
            extend_length = d

        new_node.x += extend_length * np.cos(theta)
        new_node.y += extend_length * np.sin(theta)
        return new_node

    def generate_final_path(self):
        """Tạo path từ goal về start"""
        path = [[self.goal.x, self.goal.y]]
        node = self.goal
        while node.parent is not None:
            path.append([node.parent.x, node.parent.y])
            node = node.parent
        return path[::-1]

    def get_random_node(self):
        """Sample node với goal bias"""
        if random.randint(0, 100) <= self.goal_sample_rate:
            return RRTNode(self.goal.x, self.goal.y)
        else:
            return RRTNode(
                random.uniform(0, self.grid_width),
                random.uniform(0, self.grid_height)
            )

    def is_collision_free(self, new_node, nearest_node):
        """Kiểm tra collision với tolerance"""
        check_radius = self.robot_radius * 0.9
        p1 = np.array([nearest_node.x, nearest_node.y])
        p2 = np.array([new_node.x, new_node.y])

        for obs in self.obstacles:
            if obs.static:
                obs_left = obs.x - obs.width / 2
                obs_top = obs.y - obs.height / 2

                inflated_rect = pygame.Rect(
                    obs_left - check_radius,
                    obs_top - check_radius,
                    obs.width + 2 * check_radius,
                    obs.height + 2 * check_radius
                )

                if inflated_rect.clipline(p1, p2):
                    return False

        return True

    def get_nearest_node_index(self, node_list, rnd_node):
        """Tìm node gần nhất"""
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        return dlist.index(min(dlist))

    def calc_dist_to_goal(self, x, y):
        """Tính khoảng cách đến goal"""
        return np.linalg.norm([x - self.goal.x, y - self.goal.y])

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """Tính khoảng cách và góc"""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta


# ==============================================================================
# CONTROLLER - CHỈ DÙNG RRT (ĐÃ SỬA ĐỔI)
# ==============================================================================

class IndoorAdaptedController(Controller):
    """
    Controller đơn giản: Chỉ dùng RRT để planning và follow path (đã tối ưu)
    """

    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height,
                 is_training=False, model_path=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("Initializing RRT-Only Controller (Optimized Stable)...")

        # Path planning
        self.current_path = None
        self.target_waypoint_index = 0
        self.waypoint_threshold = self.cell_size * 1.5

        # Obstacle memory
        self.known_static_obstacles = []

        # Replanning
        self.replanning_cooldown = 0
        self.last_replan_position = None

        # Stuck detection
        self.stuck_counter = 0
        self.last_position = None
        self.position_history = []

        # Previous direction for smooth movement
        self.previous_direction = (0, 0)

    def make_decision(self, robot, obstacles):
        self.replanning_cooldown = max(0, self.replanning_cooldown - 1)
        robot_pos = (robot.x, robot.y)

        static_obstacles = [obs for obs in obstacles if obs.static]
        self._update_obstacle_memory(static_obstacles)
        self._update_stuck_detection(robot_pos)

        # ------------------------------------------------------------------
        # 1. KIỂM TRA ĐIỀU KIỆN REPLAN (Ổn định hóa ngưỡng Stuck = 8)
        # ------------------------------------------------------------------
        needs_replan = False
        replan_reason = ""

        if self.current_path is None or len(self.current_path) == 0:
            needs_replan = True
            replan_reason = "No path"

        elif self.target_waypoint_index >= len(self.current_path) - 1:
            if np.hypot(robot_pos[0] - self.goal[0], robot_pos[1] - self.goal[1]) < self.waypoint_threshold:
                return (0, 0)
            needs_replan = True
            replan_reason = "Path completed"

        elif self._is_path_blocked(robot_pos, static_obstacles):
            needs_replan = True
            replan_reason = "Path blocked"

        # STUCK QUÁ 8 bước → replan ngay lập tức
        elif self.stuck_counter > 8:
            needs_replan = True
            replan_reason = f"Stuck detected ({self.stuck_counter})"
            self.stuck_counter = 0  # reset

        # ------------------------------------------------------------------
        # 2. REPLAN (Sử dụng tham số trung bình, Cooldown tăng lên 20)
        # ------------------------------------------------------------------
        if needs_replan and self.replanning_cooldown == 0:
            print(f"Replanning... ({replan_reason})")

            # Tham số trung bình khi Stuck để tránh phản ứng thái quá
            expand_dist = self.cell_size * 0.7 if "Stuck" in replan_reason else self.cell_size * 0.9
            goal_bias = 30 if "Stuck" in replan_reason else 25

            def run_rrt(start, goal, obstacles, expand_dist, goal_bias):
                planner = OptimizedRRTPlanner(
                    start=start, goal=goal, obstacles=obstacles,
                    grid_width=self.env_padding * 2 + self.grid_width * self.cell_size,
                    grid_height=self.env_padding * 2 + self.grid_height * self.cell_size,
                    robot_radius=robot.radius, expand_dist=expand_dist,
                    max_iter=6000, goal_sample_rate=goal_bias
                )
                return planner.plan()

            # Lần thử 1: Với bộ nhớ vật cản hiện tại
            new_path = run_rrt(robot_pos, self.goal, self.known_static_obstacles, expand_dist, goal_bias)

            if new_path and len(new_path) > 3:
                self.current_path = new_path
                self.target_waypoint_index = 0
                print(f"New path: {len(new_path)} waypoints")
            else:
                # Lần thử 2: Thêm chướng ngại vật tạm thời tại vị trí kẹt
                print("RRT failed → thêm temporary obstacle và thử lại")
                from pygame import Rect
                temp_obs = type('obj', (object,), {
                    'x': robot.x, 'y': robot.y,
                    'width': self.cell_size * 3, 'height': self.cell_size * 3,
                    'static': True
                })()
                temp_obstacles = self.known_static_obstacles + [temp_obs]

                new_path = run_rrt(robot_pos, self.goal, temp_obstacles, expand_dist, goal_bias)

                if new_path and len(new_path) > 3:
                    self.current_path = new_path
                    self.target_waypoint_index = 0
                    print(f"New path (Temp Obs): {len(new_path)} waypoints")
                else:
                    print("RRT failed 2 lần → đứng im.")
                    self.replanning_cooldown = 20
                    return (0, 0)

            self.replanning_cooldown = 20  # Tăng Cooldown lên 20
            self.last_replan_position = robot_pos

        # ------------------------------------------------------------------
        # 3. PATH FOLLOWING SỬ DỤNG LOOKAHEAD POINT
        # ------------------------------------------------------------------
        if not self.current_path or self.target_waypoint_index >= len(self.current_path):
            return (0, 0)

        # Cập nhật waypoint đã đến
        target_waypoint = self.current_path[self.target_waypoint_index]
        dist_to_waypoint = np.hypot(robot_pos[0] - target_waypoint[0],
                                    robot_pos[1] - target_waypoint[1])

        if dist_to_waypoint < self.waypoint_threshold:
            self.target_waypoint_index += 1
            if self.target_waypoint_index >= len(self.current_path):
                return (0, 0)

        # Chọn lookahead point
        lookahead_dist = self.cell_size * 4
        target_point = self._get_lookahead_point(robot_pos, lookahead_dist)

        dx = target_point[0] - robot_pos[0]
        dy = target_point[1] - robot_pos[1]
        desired_angle = np.arctan2(dy, dx)

        # ------------------------------------------------------------------
        # 4. CHỌN HƯỚNG DI CHUYỂN AN TOÀN
        # ------------------------------------------------------------------
        valid_moves = [(d, self._is_move_safe(robot, d, static_obstacles)) for d in self.directions]
        safe_directions = [d for d, safe in valid_moves if safe]

        if safe_directions:
            # Chọn hướng gần nhất với desired_angle
            best_dir = min(safe_directions,
                           key=lambda d: abs(self._normalize_angle(
                               np.arctan2(d[1], d[0]) - desired_angle)))
            self.previous_direction = best_dir
            return best_dir

        # ------------------------------------------------------------------
        # 5. CƠ CHẾ THOÁT HIỂM TỨC THỜI (Soft Backtracking)
        # ------------------------------------------------------------------
        else:
            # Tìm hướng ngược lại với hướng di chuyển gần nhất (hoặc ngẫu nhiên)
            if self.previous_direction != (0, 0):
                back_dir = (-self.previous_direction[0], -self.previous_direction[1])
            else:
                back_dir = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

            # Chỉ lùi nếu hướng đó an toàn
            if self._is_move_safe(robot, back_dir, static_obstacles):
                print("Lùi tức thời (Soft Backtrack) để thoát va chạm")
                # KHÔNG thay đổi self.target_waypoint_index để giữ mục tiêu phía trước
                return back_dir

                # Vẫn không được → đứng im
            print("Toàn bộ hướng bị chặn → Đứng im chờ replan")
            return (0, 0)

    def _get_lookahead_point(self, robot_pos, lookahead_dist):
        """Tính toán điểm nhìn trước trên đường đi."""
        current_node = robot_pos
        lookahead_index = self.target_waypoint_index
        total_dist = 0

        while lookahead_index < len(self.current_path):
            next_waypoint = self.current_path[lookahead_index]
            segment_dist = np.hypot(current_node[0] - next_waypoint[0], current_node[1] - next_waypoint[1])

            if total_dist + segment_dist >= lookahead_dist:
                # Lookahead point nằm trên đoạn này
                fraction = (lookahead_dist - total_dist) / segment_dist
                lookahead_x = current_node[0] + fraction * (next_waypoint[0] - current_node[0])
                lookahead_y = current_node[1] + fraction * (next_waypoint[1] - current_node[1])
                return (lookahead_x, lookahead_y)

            total_dist += segment_dist
            current_node = next_waypoint
            lookahead_index += 1

        # Nếu không đủ path, dùng waypoint cuối cùng
        return self.current_path[-1]

    def _update_stuck_detection(self, robot_pos):
        """
        Phát hiện robot bị kẹt. Đã giảm hình phạt và tăng cửa sổ kiểm tra
        để tránh phản ứng thái quá.
        """
        self.position_history.append(robot_pos)
        if len(self.position_history) > 40:
            self.position_history.pop(0)

        # 1. Phát hiện kẹt (Đi vòng/Đi loanh quanh)
        if len(self.position_history) >= 40:
            recent_positions = self.position_history[-40:]

            unique_cells = set([
                (int(p[0] / (self.cell_size * 2)), int(p[1] / (self.cell_size * 2)))
                for p in recent_positions
            ])

            # Chỉ tăng 1 điểm phạt khi ở trong 3 khu vực lớn trở xuống trong 40 bước
            if len(unique_cells) <= 3:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        # 2. Phát hiện kẹt (Di chuyển chậm)
        if self.last_position is not None:
            dist_moved = np.linalg.norm(
                np.array(robot_pos) - np.array(self.last_position)
            )

            if dist_moved < 0.1 * self.cell_size:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        self.last_position = robot_pos

    def _is_move_safe(self, robot, direction, static_obstacles):
        """Kiểm tra di chuyển có an toàn không"""
        next_x = robot.x + direction[0] * self.cell_size
        next_y = robot.y + direction[1] * self.cell_size

        # Kiểm tra biên
        if not (self.env_padding + robot.radius <= next_x <=
                self.env_padding + self.grid_width * self.cell_size - robot.radius and
                self.env_padding + robot.radius <= next_y <=
                self.env_padding + self.grid_height * self.cell_size - robot.radius):
            return False

        # Kiểm tra va chạm
        safe_radius = robot.radius * 0.9
        for obs in static_obstacles:
            if obs.static:
                obs_left = obs.x - obs.width / 2
                obs_top = obs.y - obs.height / 2
                obstacle_rect = pygame.Rect(obs_left, obs_top, obs.width, obs.height)

                if self._circle_collides_rect((next_x, next_y), safe_radius, obstacle_rect):
                    return False

        return True

    def _circle_collides_rect(self, circle_center, circle_radius, rect):
        """Kiểm tra va chạm circle-rectangle"""
        closest_x = max(rect.left, min(circle_center[0], rect.right))
        closest_y = max(rect.top, min(circle_center[1], rect.bottom))

        distance_x = circle_center[0] - closest_x
        distance_y = circle_center[1] - closest_y

        return (distance_x ** 2 + distance_y ** 2) < (circle_radius ** 2)

    def _update_obstacle_memory(self, static_obstacles):
        """Cập nhật bộ nhớ vật cản tĩnh (Chỉ lưu vật cản vật lý)"""
        for obs in static_obstacles:
            is_known = False
            for known_obs in self.known_static_obstacles:
                if abs(known_obs.x - obs.x) < 5 and abs(known_obs.y - obs.y) < 5:
                    is_known = True
                    break

            if not is_known:
                self.known_static_obstacles.append(obs)

    def _is_path_blocked(self, robot_pos, obstacles):
        """Kiểm tra path có bị chặn không"""
        if not self.current_path or len(self.current_path) < 2:
            return False

        # Kiểm tra 5 waypoints tiếp theo
        check_end = min(
            len(self.current_path) - 1,
            self.target_waypoint_index + 5
        )

        for i in range(self.target_waypoint_index, check_end):
            p1 = self.current_path[i]
            p2 = self.current_path[i + 1]

            for obs in obstacles:
                if obs.static:
                    obs_rect = pygame.Rect(
                        obs.x - obs.width / 2 - self.cell_size / 2,
                        obs.y - obs.height / 2 - self.cell_size / 2,
                        obs.width + self.cell_size,
                        obs.height + self.cell_size
                    )
                    if obs_rect.clipline(p1, p2):
                        return True

        return False

    def _normalize_angle(self, angle):
        """Chuẩn hóa góc về [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle