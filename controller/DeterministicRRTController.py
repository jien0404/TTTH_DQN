import numpy as np
import pygame
import random
from controller.Controller import Controller


# ==============================================================================
# RRT PLANNER - TỐI ƯU HÓA (Đã Thêm Seed)
# ==============================================================================

class RRTNode:
    """Nút trong cây RRT"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        self.g_score = float('inf')


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
        self.seed = None  # Biến để nhận seed từ controller

    def plan(self):
        """Thực hiện tìm đường RRT với rewiring"""
        # <<< THAY ĐỔI QUAN TRỌNG: Đặt seed ngẫu nhiên
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dist)

            if self.is_collision_free(new_node, nearest_node):
                near_indices = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indices)

                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indices)

                    if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dist:
                        if self.is_collision_free(new_node, self.goal):
                            self.goal.parent = new_node
                            raw_path = self.generate_final_path()
                            smoothed = self.smooth_path(raw_path)
                            return smoothed

        return None

    def choose_parent(self, new_node, near_indices):
        if not near_indices:
            return new_node
        costs = []
        for i in near_indices:
            near_node = self.node_list[i]
            if self.is_collision_free(new_node, near_node):
                cost = near_node.cost + self.calc_distance(near_node, new_node)
                costs.append((cost, near_node))
        if not costs:
            return new_node
        min_cost, best_parent = min(costs, key=lambda x: x[0])
        new_node.parent = best_parent
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_indices):
        for i in near_indices:
            near_node = self.node_list[i]
            if self.is_collision_free(new_node, near_node):
                new_cost = new_node.cost + self.calc_distance(new_node, near_node)
                old_cost = near_node.cost
                if new_cost < old_cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def calc_cost(self, node):
        return node.cost

    def calc_distance(self, node1, node2):
        return np.hypot(node1.x - node2.x, node1.y - node2.y)

    def find_near_nodes(self, new_node):
        r = self.expand_dist * 2.5
        return [i for i, node in enumerate(self.node_list)
                if self.calc_distance(node, new_node) <= r]

    def smooth_path(self, path):
        if len(path) <= 2:
            return path
        smoothed_path = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            last_valid_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx, -1):
                node_a = RRTNode(path[current_idx][0], path[current_idx][1])
                node_b = RRTNode(path[i][0], path[i][1])
                if self.is_collision_free(node_b, node_a):
                    last_valid_idx = i
                    break
            smoothed_path.append(path[last_valid_idx])
            current_idx = last_valid_idx
        return smoothed_path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = RRTNode(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(from_node, to_node)
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
        if random.randint(0, 100) <= self.goal_sample_rate:
            return RRTNode(self.goal.x, self.goal.y)
        else:
            return RRTNode(
                random.uniform(0, self.grid_width),
                random.uniform(0, self.grid_height)
            )

    def is_collision_free(self, new_node, nearest_node):
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
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        return dlist.index(min(dlist))

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.goal.x, y - self.goal.y])

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta


# ==============================================================================
# CONTROLLER - CHỈ DÙNG RRT (Đã Thêm Seed Cố Định)
# ==============================================================================

class IndoorAdaptedController(Controller):
    """
    Controller đơn giản: Chỉ dùng RRT để planning và follow path
    """

    def __init__(self, goal, cell_size, env_padding, grid_width, grid_height,
                 is_training=False, model_path=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("Initializing RRT-Only Controller...")

        # <<< THAY ĐỔI QUAN TRỌNG: Đặt seed cố định để đảm bảo tính nhất quán
        self.fixed_seed = 42

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
        # 1. CẢI THIỆN STUCK DETECTION + BUỘC REPLAN SỚM HƠN
        # ------------------------------------------------------------------
        needs_replan = False
        replan_reason = ""

        if self.current_path is None or len(self.current_path) == 0:
            needs_replan = True
            replan_reason = "No path"

        elif self.target_waypoint_index >= len(self.current_path) - 1:
            if np.hypot(robot_pos[0] - self.goal[0], robot_pos[1] - self.goal[1]) < self.cell_size * 2:
                return (0, 0)
            needs_replan = True
            replan_reason = "Path completed"

        elif self._is_path_blocked(robot_pos, static_obstacles):
            needs_replan = True
            replan_reason = "Path blocked"

        elif self.stuck_counter > 8:
            needs_replan = True
            replan_reason = f"Stuck detected ({self.stuck_counter})"
            self.stuck_counter = 0

        elif self.last_position is not None:
            if np.linalg.norm(np.array(robot_pos) - np.array(self.last_position)) < 1.0:
                self.stuck_counter += 2

        # ------------------------------------------------------------------
        # 2. REPLAN
        # ------------------------------------------------------------------
        if needs_replan and self.replanning_cooldown == 0:
            print(f"Replanning... ({replan_reason})")

            expand_dist = self.cell_size * 0.8 if "Stuck" in replan_reason else self.cell_size
            goal_bias = 40 if "Stuck" in replan_reason else 25
            max_iter = 4000 if "Stuck" in replan_reason else 6000

            planner = OptimizedRRTPlanner(
                start=robot_pos,
                goal=self.goal,
                obstacles=self.known_static_obstacles,
                grid_width=self.env_padding * 2 + self.grid_width * self.cell_size,
                grid_height=self.env_padding * 2 + self.grid_height * self.cell_size,
                robot_radius=robot.radius,
                expand_dist=expand_dist,
                max_iter=max_iter,
                goal_sample_rate=goal_bias
            )
            # <<< THAY ĐỔI QUAN TRỌNG: Gán seed cố định cho planner
            planner.seed = self.fixed_seed

            new_path = planner.plan()

            if new_path and len(new_path) > 3:
                self.current_path = new_path
                self.target_waypoint_index = 0
                print(f"New path: {len(new_path)} waypoints")
            else:
                print("RRT failed → Không tìm thấy đường, giữ nguyên trạng thái")

            self.replanning_cooldown = 5
            self.last_replan_position = robot_pos

        # ------------------------------------------------------------------
        # 3. PATH FOLLOWING + BACKTRACKING THÔNG MINH
        # ------------------------------------------------------------------
        if not self.current_path or self.target_waypoint_index >= len(self.current_path):
            return (0, 0)

        target_waypoint = self.current_path[self.target_waypoint_index]
        dist_to_waypoint = np.hypot(robot_pos[0] - target_waypoint[0],
                                    robot_pos[1] - target_waypoint[1])

        if dist_to_waypoint < self.waypoint_threshold:
            self.target_waypoint_index += 1
            if self.target_waypoint_index >= len(self.current_path):
                return (0, 0)
            target_waypoint = self.current_path[self.target_waypoint_index]

        dx = target_waypoint[0] - robot_pos[0]
        dy = target_waypoint[1] - robot_pos[1]
        desired_angle = np.arctan2(dy, dx)

        valid_moves = [(d, self._is_move_safe(robot, d, static_obstacles)) for d in self.directions]
        safe_directions = [d for d, safe in valid_moves if safe]

        if safe_directions:
            best_dir = min(safe_directions,
                           key=lambda d: abs(self._normalize_angle(
                               np.arctan2(d[1], d[0]) - desired_angle)))
            self.previous_direction = best_dir
            return best_dir
        else:
            if self.target_waypoint_index > 1:
                back_idx = max(1, self.target_waypoint_index - 3)
                back_point = self.current_path[back_idx]
                back_dx = back_point[0] - robot_pos[0]
                back_dy = back_point[1] - robot_pos[1]
                back_angle = np.arctan2(back_dy, back_dx)

                back_dir = min(self.directions,
                               key=lambda d: abs(self._normalize_angle(
                                   np.arctan2(d[1], d[0]) - back_angle)))

                if self._is_move_safe(robot, back_dir, static_obstacles):
                    print(f"BACKTRACKING! Lùi về waypoint {back_idx}")
                    self.target_waypoint_index = back_idx
                    return back_dir

            return (0, 0)

    def _angle_to_grid_direction(self, target_angle, robot, static_obstacles):
        valid_directions = []
        for direction in self.directions:
            if self._is_move_safe(robot, direction, static_obstacles):
                dir_angle = np.arctan2(direction[1], direction[0])
                angle_diff = abs(self._normalize_angle(target_angle - dir_angle))
                valid_directions.append((direction, angle_diff))
        if not valid_directions:
            for direction in self.directions:
                if self._is_move_safe(robot, direction, static_obstacles):
                    return direction
            return (0, 0)
        best_direction = min(valid_directions, key=lambda x: x[1])[0]
        return best_direction

    def _is_move_safe(self, robot, direction, static_obstacles):
        next_x = robot.x + direction[0] * self.cell_size
        next_y = robot.y + direction[1] * self.cell_size

        if not (self.env_padding + robot.radius <= next_x <=
                self.env_padding + self.grid_width * self.cell_size - robot.radius and
                self.env_padding + robot.radius <= next_y <=
                self.env_padding + self.grid_height * self.cell_size - robot.radius):
            return False

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
        closest_x = max(rect.left, min(circle_center[0], rect.right))
        closest_y = max(rect.top, min(circle_center[1], rect.bottom))
        distance_x = circle_center[0] - closest_x
        distance_y = circle_center[1] - closest_y
        return (distance_x ** 2 + distance_y ** 2) < (circle_radius ** 2)

    def _update_obstacle_memory(self, static_obstacles):
        for obs in static_obstacles:
            is_known = False
            for known_obs in self.known_static_obstacles:
                if abs(known_obs.x - obs.x) < 5 and abs(known_obs.y - obs.y) < 5:
                    is_known = True
                    break
            if not is_known:
                self.known_static_obstacles.append(obs)

    def _update_stuck_detection(self, robot_pos):
        self.position_history.append(robot_pos)
        if len(self.position_history) > 50:
            self.position_history.pop(0)

        if len(self.position_history) >= 30:
            recent_positions = self.position_history[-30:]
            unique_positions = set([
                (int(p[0] / self.cell_size), int(p[1] / self.cell_size))
                for p in recent_positions
            ])
            if len(unique_positions) < 5:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        if self.last_position is not None:
            dist_moved = np.linalg.norm(
                np.array(robot_pos) - np.array(self.last_position)
            )
            if dist_moved < 0.5:
                self.stuck_counter += 1

        self.last_position = robot_pos

    def _is_path_blocked(self, robot_pos, obstacles):
        if not self.current_path or len(self.current_path) < 2:
            return False

        check_end = min(
            len(self.current_path) - 1,
            self.target_waypoint_index + 5
        )

        for i in range(self.target_waypoint_index, check_end):
            p1 = self.current_path[i]
            p2 = self.current_path[i + 1]

            for obs in obstacles:
                if obs.static:
                    safe_margin = self.cell_size / 3
                    obs_rect = pygame.Rect(
                        obs.x - obs.width / 2 - safe_margin,
                        obs.y - obs.height / 2 - safe_margin,
                        obs.width + 2 * safe_margin,
                        obs.height + 2 * safe_margin
                    )
                    if obs_rect.clipline(p1, p2):
                        return True
        return False

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle