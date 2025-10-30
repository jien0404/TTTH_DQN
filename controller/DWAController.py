import numpy as np
from controller.Controller import Controller


class DWAController(Controller):
    """Dynamic Window Approach (DWA) controller for robot path planning, non-learning algorithm."""
    
    def __init__(self, goal, cell_size, env_padding, is_training=False, model_path=None, max_speed_ratio=0.5, max_turn_rate=np.pi):
        """
        Initialize the DWA controller.

        Args:
            goal (tuple): Target position (x, y) to navigate towards.
            cell_size (float): Size of each grid cell.
            env_padding (float): Padding around the environment boundaries.
            is_training (bool, optional): Training mode (always False for DWA). Defaults to False.
            model_path (str, optional): Path to model (not used for DWA). Defaults to None.
            max_speed_ratio (float, optional): Maximum linear speed ratio. Defaults to 0.5.
            max_turn_rate (float, optional): Maximum angular speed (rad/s). Defaults to np.pi.
        """
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self.max_speed_ratio = max_speed_ratio
        self.max_turn_rate = max_turn_rate

    def _initialize_algorithm(self):
        """Initialize algorithm-specific components (none for DWA)."""
        pass

    def make_decision(self, robot, obstacles):
        """
        Make a control decision considering obstacles.

        Args:
            robot: Robot object with position (robot.pos) and vision (robot.vision).
            obstacles: List of Obstacle objects.

        Returns:
            tuple: Direction (dx, dy) from self.directions.
        """
        # Prepare obstacle positions in format [x, y, width, height]
        obstacle_positions = [obs.get_bounding_box() for obs in obstacles]
        
        # Generate feasible velocity commands
        valid_speeds = np.linspace(0, self.max_speed_ratio, num=4)
        valid_turn_rates = np.linspace(-self.max_turn_rate, self.max_turn_rate, num=9)

        # Initialize best commands
        best_cmd = (0, 0)
        best_score = float('-inf')
        robot_pose = (robot.x, robot.y)

        # Iterate through feasible velocity commands
        for v in valid_speeds:
            for w in valid_turn_rates:
                # Simulate motion
                simulated_trajectory = self.simulate_motion(robot_pose, v, w)

                # Evaluate trajectory
                score = self.evaluate_trajectory(simulated_trajectory, obstacle_positions, robot.vision * 0.75)

                # Update best commands if score is better
                if score > best_score:
                    best_score = score
                    best_cmd = (v, w)

        # Convert best command to direction (dx, dy)
        direction = (best_cmd[0] * np.cos(best_cmd[1]), best_cmd[0] * np.sin(best_cmd[1]))

        # Map to closest discrete direction
        closest_direction = min(self.directions, key=lambda d: np.linalg.norm(np.array(d) - np.array(direction)))
        return closest_direction

    def simulate_motion(self, pose, v, w):
        """
        Simulate robot motion given current pose and velocity commands.

        Args:
            pose (tuple): Current robot position (x, y).
            v (float): Linear velocity.
            w (float): Angular velocity (used as orientation angle).

        Returns:
            list: Simulated next position [x, y].
        """
        new_x = pose[0] + v * np.cos(w) * self.cell_size
        new_y = pose[1] + v * np.sin(w) * self.cell_size
        return [new_x, new_y]

    def evaluate_trajectory(self, trajectory, obstacles, safety_distance):
        """
        Evaluate a trajectory based on distance to goal and obstacle avoidance.

        Args:
            trajectory (list): Simulated position [x, y].
            obstacles (list): List of obstacles as [x_min, x_max, y_min, y_max].
            safety_distance (float): Minimum safe distance to obstacles.

        Returns:
            float: Score of the trajectory.
        """
        distance_to_goal = np.linalg.norm(np.array(trajectory) - np.array(self.goal))
        score = 1.0 / (distance_to_goal + 1)  # Penalize distance to goal
        for obstacle in obstacles:
            closest_x = max(obstacle[0], min(trajectory[0], obstacle[1]))
            closest_y = max(obstacle[2], min(trajectory[1], obstacle[3]))
            dist_to_obstacle = ((closest_x - trajectory[0]) ** 2 + (closest_y - trajectory[1]) ** 2) ** 0.5
            if dist_to_obstacle < safety_distance:
                score -= 10.0
        return score

    def store_experience(self, state, action_idx, reward, next_state, done):
        """
        Store experience in memory (not applicable for DWA).

        Args:
            state: Current state.
            action_idx: Action index.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        pass  # DWA does not use experience replay

    def train(self):
        """Perform a training step (not applicable for DWA)."""
        pass  # DWA does not train

    def update_epsilon(self):
        """Update exploration rate (not applicable for DWA)."""
        pass  # DWA does not use exploration

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        """
        Calculate reward for the current state.

        Args:
            robot: Robot object.
            obstacles: List of Obstacle objects.
            done: Whether the episode is done.
            reached_goal: Whether the goal was reached.
            distance_to_goal: Current distance to goal.
            prev_distance: Previous distance to goal.

        Returns:
            float: Calculated reward.
        """
        if reached_goal:
            return 100.0  # Large positive reward for reaching goal
        if robot.check_collision(obstacles):
            return -100.0  # Large negative reward for collision
        if prev_distance is not None and distance_to_goal < prev_distance:
            return 10.0  # Positive reward for moving closer to goal
        return -1.0  # Small negative reward for each step

    def _save_model_implementation(self):
        """Implement model saving (not applicable for DWA)."""
        print("DWAController does not use a model to save.")

    def _load_model_implementation(self):
        """Implement model loading (not applicable for DWA)."""
        print("DWAController does not use a model to load.")