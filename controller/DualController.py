import numpy as np
import torch
import os
from collections import defaultdict
from Controller import Controller
import torch.serialization  # Added for safe_globals
import builtins  # Added for float
from numpy._core.multiarray import scalar  # Added for numpy scalar

# Allow defaultdict, float, and numpy scalar to be loaded safely with weights_only=True
torch.serialization.add_safe_globals([defaultdict, builtins.float, scalar])

class DualQLController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="dualql_model.pth"):
        # Đảm bảo các thuộc tính được định nghĩa trước khi gọi hàm khởi tạo lớp cha
        # Định nghĩa action_space trước tiên
        self.action_space = [
            "up_1", "down_1", "left_1", "right_1",
            "up-left_1", "up-right_1", "down-left_1", "down-right_1"
        ]

        # Sau đó mới gọi hàm khởi tạo lớp cha
        super().__init__(goal, cell_size, env_padding, is_training, model_path)

        # Hyperparameters from DualQL.py
        self.ALPHA = 0.9
        self.GAMMA = 0.9
        self.EPSILON = 0.5 if is_training else 0.0
        self.EPSILON_DECAY = 0.95
        self.SUCCESS_REWARD = 15
        self.COLLISION_DISCOUNT = -5
        self.OBSTACLE_REWARD_FACTOR = 0.2
        self.EPSILON_MIN = 0.01

        # Map action_space to self.directions
        self.action_to_direction = {
            "up_1": (0, -1), "down_1": (0, 1), "left_1": (-1, 0), "right_1": (1, 0),
            "up-left_1": (-1, -1), "up-right_1": (1, -1),
            "down-left_1": (-1, 1), "down-right_1": (1, 1)
        }
        # Map self.directions to action_space
        self.direction_to_action = {v: k for k, v in self.action_to_direction.items()}

        # Discrete values for obstacle state
        self.DESCRETE_PHI = [30, 60]
        self.DESCRETE_DELTAPHI = [-4, -2, 2, 4]
        self.DESCRETE_DELTAD = [-1, 1]

        # Initialize Q-tables and policies
        def default_policy():
            return "right_1"  # Default action if state not initialized
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.obstacle_q_table = defaultdict(lambda: defaultdict(float))
        self.policy = defaultdict(default_policy)
        self.obstacle_policy = defaultdict(default_policy)
        self.episode_decisions = []
        self.obstacle_episode_decisions = []
        self.metrics = {
            'sum_rewards': [],
            'average_rewards': []
        }

        # Initialize policies
        self._initialize_policies()

        # Load model if not training
        if not is_training:
            self.load_model()
    
    def _initialize_policies(self):
        """Initialize Q-tables and policies as in DualQL.py"""
        grid_size = int((512 - 2 * self.env_padding) / self.cell_size)
        goal_grid_x = int((self.goal[0] - self.env_padding) / self.cell_size)
        goal_grid_y = int((self.goal[1] - self.env_padding) / self.cell_size)

        # Initialize normal Q-table and policy
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                distance = np.sqrt((i - goal_grid_x)**2 + (j - goal_grid_y)**2)
                initial_value = self.cell_size / max(distance, 1)
                for action in self.action_space:
                    self.q_table[state][action] = initial_value
                initial_action = self._get_initial_decision(state, (goal_grid_x, goal_grid_y))
                self.policy[state] = initial_action

        # Initialize obstacle Q-table and policy
        for c_phi in range(3):  # 0, 1, 2
            for c_deltaphi in range(-2, 3):  # -2, -1, 0, 1, 2
                for c_deltad in range(-1, 2):  # -1, 0, 1
                    for goal_direction in range(8):  # 0-7
                        state = (c_phi, c_deltaphi, c_deltad, goal_direction)
                        for action in self.action_space:
                            self.obstacle_q_table[state][action] = 0
                        action_map = {
                            0: "right_1", 1: "up-right_1", 2: "up_1", 3: "up-left_1",
                            4: "left_1", 5: "down-left_1", 6: "down_1", 7: "down-right_1"
                        }
                        self.obstacle_policy[state] = action_map[goal_direction]

    def _initialize_state(self, state, goal_grid_x, goal_grid_y):
        """Initialize Q-table and policy for a new state"""
        if state is None:
            return  # Skip initialization if state is None
        distance = np.sqrt((state[0] - goal_grid_x)**2 + (state[1] - goal_grid_y)**2)
        initial_value = self.cell_size / max(distance, 1)
        for action in self.action_space:
            self.q_table[state][action] = initial_value
        initial_action = self._get_initial_decision(state, (goal_grid_x, goal_grid_y))
        self.policy[state] = initial_action

    def _get_initial_decision(self, state, goal):
        """Calculate initial action based on direction to goal"""
        x, y = state
        goal_x, goal_y = goal
        relative_x = goal_x - x
        relative_y = goal_y - y
        tan_pi_8 = np.tan(np.pi / 8)

        if relative_x >= 0:
            if relative_y >= 0:
                if relative_y == 0 or relative_x / relative_y > 1 / tan_pi_8:
                    return "right_1"
                elif relative_x == 0 or relative_x / relative_y < tan_pi_8:
                    return "up_1"
                else:
                    return "up-right_1"
            else:
                if relative_y == 0 or -relative_x / relative_y < tan_pi_8:
                    return "right_1"
                elif relative_x == 0 or -relative_x / relative_y > 1 / tan_pi_8:
                    return "down_1"
                else:
                    return "down-right_1"
        else:
            if relative_y >= 0:
                if relative_y == 0 or relative_x / relative_y < -1 / tan_pi_8:
                    return "left_1"
                elif relative_x == 0 or relative_x / relative_y > -tan_pi_8:
                    return "up_1"
                else:
                    return "up-left_1"
            else:
                if relative_y == 0 or relative_x / relative_y > -tan_pi_8:
                    return "left_1"
                elif relative_x == 0 or relative_x / relative_y < -1 / tan_pi_8:
                    return "down_1"
                else:
                    return "down-left_1"
    def _convertphi(self, phi):
        """Convert phi (degrees) to discrete state"""
        if phi < self.DESCRETE_PHI[0]:
            return 0
        elif phi < self.DESCRETE_PHI[1]:
            return 1
        else:
            return 2


    def _convertdeltaphi(self, deltaphi):
        """Convert delta_phi (degrees) to discrete state"""
        if deltaphi < self.DESCRETE_DELTAPHI[0]:
            return -2
        elif deltaphi < self.DESCRETE_DELTAPHI[1]:
            return -1
        elif deltaphi < self.DESCRETE_DELTAPHI[2]:
            return 0
        elif deltaphi < self.DESCRETE_DELTAPHI[3]:
            return 1
        else:
            return 2

    def _convertdeltad(self, deltad):
        """Convert delta_d to discrete state"""
        if deltad < self.DESCRETE_DELTAD[0]:
            return -1
        elif deltad < self.DESCRETE_DELTAD[1]:
            return 0
        else:
            return 1

    def _angle(self, x1, y1, x2, y2):
        """Calculate angle between two vectors"""
        return np.arccos((x1 * x2 + y1 * y2) / (np.sqrt(x1**2 + y1**2) * np.sqrt(x2**2 + y2**2) + 1e-6))

    def _find_octant(self, x, y, goal):
        """Determine goal direction (0-7)"""
        relative_x = goal[0] - x
        relative_y = goal[1] - y
        if relative_x >= 0:
            if relative_y >= 0:
                if relative_x >= relative_y:
                    return 0
                else:
                    return 1
            else:
                if relative_x >= -relative_y:
                    return 7
                else:
                    return 6
        else:
            if relative_y >= 0:
                if -relative_x >= relative_y:
                    return 3
                else:
                    return 2
            else:
                if relative_x <= relative_y:
                    return 4
                else:
                    return 5

    def _remap_keys(self, mapping):
        """Convert dictionary to list of key-value pairs for JSON"""
        return [{'key': k, 'value': v} for k, v in mapping.items()]

    def _get_obstacle_state(self, robot, obstacles):
        """Calculate obstacle state (phi, delta_phi, delta_d, goal_direction) and return closest obstacle"""
        moving_obstacles = robot.detect_moving_obstacles(obstacles)
        if not moving_obstacles:
            return None, None

        closest_obstacle = moving_obstacles[0]
        obs = closest_obstacle["obstacle"]
        distance = closest_obstacle["distance"]
        obs_x, obs_y = obs.x, obs.y

        # Store current state to restore later
        original_x, original_y = obs.x, obs.y
        original_history = obs.history.copy()

        # Calculate next position of obstacle
        obs.move()
        distance_next = ((obs.x - robot.x)**2 + (obs.y - robot.y)**2)**0.5
        obs_x_next, obs_y_next = obs.x, obs.y

        # Restore obstacle state
        obs.x, obs.y = original_x, original_y
        obs.history = original_history

        # Calculate phi and phi_next
        goal_direction = (self.goal[0] - robot.x, self.goal[1] - robot.y)
        obstacle_direction = (obs_x - robot.x, obs_y - robot.y)
        obstacle_direction_next = (obs_x_next - robot.x, obs_y_next - robot.y)
        phi = self._angle(goal_direction[0], goal_direction[1], obstacle_direction[0], obstacle_direction[1])
        phi_next = self._angle(goal_direction[0], goal_direction[1], obstacle_direction_next[0], obstacle_direction_next[1])

        # Convert to discrete states
        c_phi = self._convertphi(phi / np.pi * 180)
        c_deltaphi = self._convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = self._convertdeltad(distance_next - distance)
        goal_direction_idx = self._find_octant(robot.grid_x, robot.grid_y, (
            int((self.goal[0] - self.env_padding) / self.cell_size),
            int((self.goal[1] - self.env_padding) / self.cell_size)
        ))

        return (c_phi, c_deltaphi, c_deltad, goal_direction_idx), closest_obstacle

    def make_decision(self, robot, obstacles):
        """Make decision based on state and policy, store current state"""
        state = (robot.grid_x, robot.grid_y)
        obstacle_state, _ = self._get_obstacle_state(robot, obstacles)

        if obstacle_state and self.is_training:
            # Obstacle decision
            if np.random.random() < self.EPSILON:
                action = np.random.choice(self.action_space)
            else:
                action = self.obstacle_policy[obstacle_state]
                if action not in self.action_space:
                    action = np.random.choice(self.action_space)
            self.obstacle_episode_decisions.append((state, action, 0, obstacle_state, None))
        else:
            # Normal decision
            if np.random.random() < self.EPSILON and self.is_training:
                action = np.random.choice(self.action_space)
            else:
                action = self.policy[state]
                if action not in self.action_space:
                    action = np.random.choice(self.action_space)
            self.episode_decisions.append((state, action, 0, None, None))

        return self.action_to_direction[action]

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        """Calculate reward as in DualQL.py, store next state"""
        state = (robot.grid_x, robot.grid_y)
        weight = 1.0 if self.direction_to_action.get(robot.direction, "").endswith("_1") in [
            "up_1", "down_1", "left_1", "right_1"
        ] else 1.0 / np.sqrt(2)

        obstacle_state, closest_obstacle = self._get_obstacle_state(robot, obstacles)
        if obstacle_state and closest_obstacle:
            # Obstacle reward
            distance_to_obstacle = closest_obstacle["distance"]
            obs = closest_obstacle["obstacle"]

            # Store current state
            original_x, original_y = obs.x, obs.y
            original_history = obs.history.copy()

            # Calculate distance after move
            obs.move()
            distance_to_obstacle_after = ((obs.x - robot.x)**2 + (obs.y - robot.y)**2)**0.5

            # Restore obstacle state
            obs.x, obs.y = original_x, original_y
            obs.history = original_history

            reward = (
                ((prev_distance - distance_to_goal) / max(np.abs(prev_distance - distance_to_goal), 1e-6)) * weight +
                self.OBSTACLE_REWARD_FACTOR * (
                    (distance_to_obstacle_after - distance_to_obstacle) /
                    max(np.abs(distance_to_obstacle_after - distance_to_obstacle), 1e-6)
                ) * weight
            )
            if self.obstacle_episode_decisions:
                last_state, last_action, _, last_obstacle_state, _ = self.obstacle_episode_decisions[-1]
                self.obstacle_episode_decisions[-1] = (last_state, last_action, reward, last_obstacle_state, state)
        else:
            # Normal reward
            reward = ((prev_distance - distance_to_goal) / max(np.abs(prev_distance - distance_to_goal), 1e-6)) * weight
            if self.episode_decisions:
                last_state, last_action, _, last_obstacle_state, _ = self.episode_decisions[-1]
                self.episode_decisions[-1] = (last_state, last_action, reward, last_obstacle_state, state)

        if reached_goal:
            reward += self.SUCCESS_REWARD
            if self.episode_decisions:
                last_state, last_action, _, last_obstacle_state, _ = self.episode_decisions[-1]
                self.episode_decisions[-1] = (last_state, last_action, reward, last_obstacle_state, state)
            if self.obstacle_episode_decisions:
                last_state, last_action, _, last_obstacle_state, _ = self.obstacle_episode_decisions[-1]
                self.obstacle_episode_decisions[-1] = (last_state, last_action, reward, last_obstacle_state, state)
        elif robot.check_collision(obstacles):
            reward += self.COLLISION_DISCOUNT
            if self.episode_decisions:
                last_state, last_action, _, last_obstacle_state, _ = self.episode_decisions[-1]
                self.episode_decisions[-1] = (last_state, last_action, reward, last_obstacle_state, state)
            if self.obstacle_episode_decisions:
                last_state, last_action, _, last_obstacle_state, _ = self.obstacle_episode_decisions[-1]
                self.obstacle_episode_decisions[-1] = (last_state, last_action, reward, last_obstacle_state, state)

        return reward
    def train(self):
        """Update Q-tables and policies"""
        goal_grid_x = int((self.goal[0] - self.env_padding) / self.cell_size)
        goal_grid_y = int((self.goal[1] - self.env_padding) / self.cell_size)

        # Update normal Q-table
        for decision in self.episode_decisions:
            if len(decision) == 4:  # Handle old format
                state, action, reward, _ = decision
                next_state = state  # Fallback to current state
            else:  # Expected format
                state, action, reward, _, next_state = decision
            if next_state is None:
                next_state = state  # Use current state if next_state is None
            if not self.q_table[next_state]:
                self._initialize_state(next_state, goal_grid_x, goal_grid_y)
            best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
            self.q_table[state][action] = (
                (1 - self.ALPHA) * self.q_table[state][action] +
                self.ALPHA * (reward + self.GAMMA * self.q_table[next_state][best_next_action])
            )
            self.policy[state] = max(self.q_table[state], key=self.q_table[state].get)

        # Update obstacle Q-table
        for decision in self.obstacle_episode_decisions:
            if len(decision) == 4:  # Handle old format
                state, action, reward, obstacle_state = decision
            else:  # Expected format
                state, action, reward, obstacle_state, _ = decision
            if obstacle_state:
                best_next_action = max(self.obstacle_q_table[obstacle_state], key=self.obstacle_q_table[obstacle_state].get)
                self.obstacle_q_table[obstacle_state][action] = (
                    (1 - self.ALPHA) * self.obstacle_q_table[obstacle_state][action] +
                    self.ALPHA * (reward + self.GAMMA * self.obstacle_q_table[obstacle_state][best_next_action])
                )
                self.obstacle_policy[obstacle_state] = max(self.obstacle_q_table[obstacle_state], key=self.obstacle_q_table[obstacle_state].get)

        # Update metrics
        sum_rewards = sum(r for _, _, r, *_ in self.episode_decisions) + sum(r for _, _, r, *_ in self.obstacle_episode_decisions)
        self.metrics['sum_rewards'].append(sum_rewards)
        self.metrics['average_rewards'].append(sum_rewards / max(1, len(self.episode_decisions) + len(self.obstacle_episode_decisions)))

        # Clear decisions
        self.episode_decisions.clear()
        self.obstacle_episode_decisions.clear()

    def store_experience(self, state, action_idx, reward, next_state, done):
        """Store experience (not used for Q-table, handled by episode_decisions)"""
        pass

    def save_model(self):
        """Save Q-tables, policies, and metrics using PyTorch"""
        model_data = {
            'q_table': dict(self.q_table),
            'obstacle_q_table': dict(self.obstacle_q_table),
            'policy': dict(self.policy),
            'obstacle_policy': dict(self.obstacle_policy),
            'epsilon': self.EPSILON,
            'metrics': self.metrics
        }
        # Đảm bảo lưu với weights_only=False để tương thích khi tải lại
        torch.save(model_data, self.model_path, pickle_protocol=4, _use_new_zipfile_serialization=False)
        print(f"Model saved to {self.model_path}")
        
    def load_model(self):
        """Load Q-tables, policies, and metrics"""
        if os.path.exists(self.model_path):
            try:
                # Thêm tham số weights_only=False để khắc phục lỗi trong PyTorch 2.6
                model_data = torch.load(self.model_path, weights_only=False)
                self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
                self.obstacle_q_table = defaultdict(lambda: defaultdict(float), model_data['obstacle_q_table'])
                self.policy = defaultdict(lambda: "right_1", model_data['policy'])
                self.obstacle_policy = defaultdict(lambda: "right_1", model_data['obstacle_policy'])
                self.EPSILON = model_data['epsilon']
                self.metrics = model_data['metrics']
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating a new model...")
                # Đảm bảo action_space đã được định nghĩa trước khi gọi _initialize_policies
                if not hasattr(self, 'action_space'):
                    self.action_space = [
                        "up_1", "down_1", "left_1", "right_1",
                        "up-left_1", "up-right_1", "down-left_1", "down-right_1"
                    ]
                # Khởi tạo lại chính sách
                self._initialize_policies()
        else:
            raise FileNotFoundError(f"Model file {self.model_path} not found!")