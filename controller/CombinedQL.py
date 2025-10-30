import numpy as np
import torch
import random
import json
import os
from controller.Controller import Controller

class CombinedQLController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="models/combined_ql_model.json"):
        """
        Initialize the CombinedQL controller.
        
        Args:
            goal: The goal position (x, y)
            cell_size: Size of each grid cell
            env_padding: Environment padding
            is_training: Whether we're in training mode
            model_path: Path to save/load model
        """
        
        # QL specific parameters
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.5 if is_training else 0.0  # Exploration rate
        self.epsilon_decay = 0.95
        self.alpha = 0.9  # Learning rate
        self.learning_rate_decay = 1.0
        
        # Reward parameters
        self.collision_penalty = -5
        self.success_reward = 15
        self.obstacle_reward_factor = 0.2
        
        # Episode tracking
        self.episode_decisions = []
        self.sum_of_rewards = []
        self.average_reward = []
        
        # Action space for the agent
        self.action_space = [
            "up_1", "down_1", "left_1", "right_1", 
            "up-right_1", "up-left_1", "down-right_1", "down-left_1"
        ]
        
        # Movement mapping
        self.decision_movement = {
            "up_1": (0, -1),
            "down_1": (0, 1),
            "left_1": (-1, 0),
            "right_1": (1, 0),
            "up-right_1": (1, -1),
            "up-left_1": (-1, -1),
            "down-right_1": (1, 1),
            "down-left_1": (-1, 1)
        }

        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
    def _initialize_algorithm(self):
        """Initialize the Q-learning algorithm."""
        self.qtable = {}
        self.policy = {}
        
        # Initialize Qtable and policy
        grid_width = int((self.goal[0] * 2 - self.env_padding) // self.cell_size)
        grid_height = int((self.goal[1] * 2 - self.env_padding) // self.cell_size)
        
        for i in range(grid_width):
            for j in range(grid_height):
                # Initialize policy to always go to the goal, wherever the robot is
                cell_center = (self.env_padding + self.cell_size / 2 + i * self.cell_size, 
                             self.env_padding + self.cell_size / 2 + j * self.cell_size)
                direction = (self.goal[0] - cell_center[0], self.goal[1] - cell_center[1])
                decision = ""

                # A 90-degree region is divided into 3 parts
                # For example, 0 - 90: right, up-right, up
                ratio = np.tan(np.pi / 8)
                if abs(direction[1]) > ratio * abs(direction[0]):
                    if direction[1] > 0:
                        decision += "down"
                    else:
                        decision += "up"

                    if abs(direction[0]) > ratio * abs(direction[1]):
                        if direction[0] > 0:
                            decision += "-right"
                        else:
                            decision += "-left"
                else:
                    if direction[0] > 0:
                        decision += "right"
                    else:
                        decision += "left"

                # Add movement magnitude
                decision += "_1"

                # Calculate distance to goal
                distance = self._calculate_distance_to_goal((i, j))
                
                # Initialize Q-values
                ini_value = self.cell_size / distance if distance > 0 else 0

                # Standard state
                self.policy[(i, j, -10, -10, -10)] = decision
                for action in self.action_space:
                    self.qtable[(i, j, -10, -10, -10, action)] = ini_value
                
                # States with obstacle information
                for phi in range(3):  # Phi: F, S, D
                    for delta_phi in range(-2, 3):  # DeltaPhi: C, LC, U, LA, A
                        for delta_d in range(-1, 2):  # DeltaD: C, U, A
                            self.policy[(i, j, phi, delta_phi, delta_d)] = decision
                            for action in self.action_space:
                                self.qtable[(i, j, phi, delta_phi, delta_d, action)] = ini_value
    
    def make_decision(self, robot, obstacles):
        """
        Make a decision based on the current state.
        
        Args:
            robot: The robot object
            obstacles: List of obstacles
            
        Returns:
            Direction as (x, y) tuple
        """
        # First update Q-values from previous step
        self._update_all()
        
        # Get current state
        state = self._convert_state(robot)
        base_state = (state[0], state[1], -10, -10, -10)  # Default state with no obstacle info
        
        # Check for nearby obstacles
        moving_obstacles = robot.detect_moving_obstacles(obstacles)
        if moving_obstacles and self.is_training:  # If there are moving obstacles and we're training
            return self._make_obstacle_decision(robot, moving_obstacles[0]["obstacle"])
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon and self.is_training:
            decision = random.choice(self.action_space)
        else:
            decision = self.policy.get(base_state, self.action_space[0])  # Default to first action if state not found
        
        # Calculate reward for this step
        distance = self._calculate_distance_to_goal(base_state)
        movement = self.decision_movement[decision]
        next_state = (state[0] + movement[0], state[1] + movement[1])
        next_distance = self._calculate_distance_to_goal(next_state)
        
        # Calculate movement weight
        if decision in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1
        else:
            weight = 1 / np.sqrt(2)
        
        # Calculate reward
        reward = ((distance - next_distance) / max(abs(distance - next_distance), 1e-6)) * weight
        
        # Store this decision for training
        if self.is_training:
            self.episode_decisions.append((base_state, decision, reward))
        
        return self.decision_movement[decision]
    
    def _make_obstacle_decision(self, robot, obstacle):
        """
        Make a decision considering obstacle avoidance.
        
        Args:
            robot: The robot object
            obstacle: The obstacle to avoid
            
        Returns:
            Direction as (x, y) tuple
        """
        # Get obstacle position before and after its movement
        obstacle_before = (obstacle.x, obstacle.y)
        # Predict next position based on obstacle velocity
        obstacle_after = (obstacle.x + getattr(obstacle, 'velocity', (0, 0))[0], 
                         obstacle.y + getattr(obstacle, 'velocity', (0, 0))[1])
        
        # Calculate distances
        distance_to_obstacle = np.sqrt((robot.x - obstacle_before[0]) ** 2 + (robot.y - obstacle_before[1]) ** 2)
        distance_to_obstacle_next = np.sqrt((robot.x - obstacle_after[0]) ** 2 + (robot.y - obstacle_after[1]) ** 2)
        
        # Calculate angles for relative position
        robot_direction = (self.goal[0] - robot.x, self.goal[1] - robot.y)
        
        # Calculate phi (angle between robot-goal and robot-obstacle)
        phi = self._angle(robot_direction[0], robot_direction[1], 
                          obstacle_before[0] - robot.x, obstacle_before[1] - robot.y)
        phi_next = self._angle(robot_direction[0], robot_direction[1], 
                              obstacle_after[0] - robot.x, obstacle_after[1] - robot.y)
        
        # Convert to state representation
        c_phi = self._convert_phi(phi / np.pi * 180)
        c_deltaphi = self._convert_deltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = self._convert_deltad((distance_to_obstacle_next - distance_to_obstacle))
        
        # Get base state
        state = self._convert_state(robot)
        obstacle_state = (state[0], state[1], c_phi, c_deltaphi, c_deltad)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            decision = random.choice(self.action_space)
        else:
            decision = self.policy.get(obstacle_state, self.action_space[0])
        
        # Calculate reward
        distance = self._calculate_distance_to_goal(state)
        movement = self.decision_movement[decision]
        next_state = (state[0] + movement[0], state[1] + movement[1])
        next_distance = self._calculate_distance_to_goal(next_state)
        
        # Calculate obstacle distance after movement
        distance_to_obstacle_after_movement = np.sqrt(
            (robot.x + movement[0] * self.cell_size - obstacle_after[0]) ** 2 + 
            (robot.y + movement[1] * self.cell_size - obstacle_after[1]) ** 2
        )
        
        # Calculate weight
        if decision in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1
        else:
            weight = 1 / np.sqrt(2)
        
        # Calculate combined reward
        goal_reward = ((distance - next_distance) / max(abs(distance - next_distance), 1e-6)) * weight
        obstacle_reward = (self.obstacle_reward_factor * 
                          (distance_to_obstacle_after_movement - distance_to_obstacle) / 
                          max(abs(distance_to_obstacle_after_movement - distance_to_obstacle), 1e-6) * weight)
                          
        reward = goal_reward + obstacle_reward
        
        # Store this decision for training
        if self.is_training:
            self.episode_decisions.append((obstacle_state, decision, reward))
        
        return self.decision_movement[decision]
    
    def set_collision(self):
        """Handle collision event for learning."""
        if self.is_training and len(self.episode_decisions) > 0:
            state, decision, reward = self.episode_decisions[-1]
            reward += self.collision_penalty
            
            self.episode_decisions[-1] = (state, decision, reward)
            self._update_all()
            self._calculate_reward()
            self.episode_decisions.clear()
    
    def set_success(self):
        """Handle goal reached event for learning."""
        if self.is_training:
            self.update_epsilon()  # Decay epsilon
            
            if len(self.episode_decisions) > 0:
                state, decision, reward = self.episode_decisions[-1]
                reward += self.success_reward
                
                self.episode_decisions[-1] = (state, decision, reward)
                
                # Add final goal state
                goal_pos = (int((self.goal[0] - self.env_padding) / self.cell_size),
                            int((self.goal[1] - self.env_padding) / self.cell_size))
                self.episode_decisions.append(((goal_pos[0], goal_pos[1], -10, -10, -10), "", 0))
                
                self._update_all()
                self._calculate_reward()
                self.episode_decisions.clear()
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer."""
        # This method is for compatibility, but we handle experience differently
        # in the Q-learning approach
        pass
    
    def train(self):
        """Train the model."""
        # Training happens incrementally as decisions are made
        pass
    
    def update_epsilon(self):
        """Update exploration rate."""
        self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        """Calculate reward for the current state."""
        if reached_goal:
            return self.success_reward
        elif done:  # Collision
            return self.collision_penalty
        
        # Distance-based reward
        distance_reward = 0
        if prev_distance is not None:
            distance_reward = prev_distance - distance_to_goal
        
        # Obstacle avoidance reward
        obstacle_reward = 0
        nearest_obstacle_dist = robot.distance_to_nearest_obstacle(obstacles)
        if nearest_obstacle_dist < robot.vision:
            obstacle_reward = self.obstacle_reward_factor * nearest_obstacle_dist / robot.vision
        
        return distance_reward + obstacle_reward
    
    def _save_model_implementation(self):
        """Save the model to disk."""
        model_data = {
            'qtable': {str(k): v for k, v in self.qtable.items()},
            'policy': {str(k): v for k, v in self.policy.items()},
            'metrics': {
                'sum_of_rewards': self.sum_of_rewards,
                'average_reward': self.average_reward
            }
        }
        
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def _load_model_implementation(self):
        """Load the model from disk."""
        with open(self.model_path, 'r') as f:
            model_data = json.load(f)
            
        # Parse string keys back to tuples
        self.qtable = {}
        for key_str, value in model_data['qtable'].items():
            key = eval(key_str)  # Convert string representation back to tuple
            self.qtable[key] = value
            
        self.policy = {}
        for key_str, value in model_data['policy'].items():
            key = eval(key_str)  # Convert string representation back to tuple
            self.policy[key] = value
            
        if 'metrics' in model_data:
            self.sum_of_rewards = model_data['metrics'].get('sum_of_rewards', [])
            self.average_reward = model_data['metrics'].get('average_reward', [])
    
    def _calculate_reward(self):
        """Calculate total and average reward for the episode."""
        if not self.episode_decisions:
            return
            
        sum_of_reward = sum(decision[2] for decision in self.episode_decisions)
        self.sum_of_rewards.append(sum_of_reward)
        self.average_reward.append(sum_of_reward / len(self.episode_decisions))
    
    def _update_qtable(self, state, decision, reward, next_state):
        """Update Q-value for a state-action pair."""
        # Find optimal value of next state
        next_q_values = [self.qtable.get((next_state[0], next_state[1], next_state[2], 
                                         next_state[3], next_state[4], action), 0) 
                         for action in self.action_space]
        optimal_q_next = max(next_q_values)
        
        # Get current Q-value
        current_q = self.qtable.get((state[0], state[1], state[2], state[3], state[4], decision), 0)
        
        # Update Q-value
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * optimal_q_next)
        self.qtable[(state[0], state[1], state[2], state[3], state[4], decision)] = new_q
    
    def _update_policy(self, state):
        """Update policy for a state based on Q-values."""
        best_action = max(self.action_space, 
                         key=lambda action: self.qtable.get((state[0], state[1], state[2], 
                                                           state[3], state[4], action), 0))
        self.policy[state] = best_action
    
    def _update_all(self):
        """Update Q-values and policy for stored experience."""
        if len(self.episode_decisions) >= 2:
            state, decision, reward = self.episode_decisions[-2]
            next_state = self.episode_decisions[-1][0]
            
            # Update Q-table
            self._update_qtable(state, decision, reward, next_state)
            
            # Update policy
            self._update_policy(state)
    
    def _convert_state(self, robot):
        """Convert robot position to grid state."""
        grid_x = int((robot.x - self.env_padding) // self.cell_size)
        grid_y = int((robot.y - self.env_padding) // self.cell_size)
        return (grid_x, grid_y, -10, -10, -10)  # Default state with no obstacle info
    
    def _calculate_distance_to_goal(self, state):
        """Calculate Euclidean distance from state to goal."""
        goal_grid_x = int((self.goal[0] - self.env_padding) // self.cell_size)
        goal_grid_y = int((self.goal[1] - self.env_padding) // self.cell_size)
        
        return ((state[0] - goal_grid_x)**2 + (state[1] - goal_grid_y)**2) ** 0.5
    
    # Helper functions for obstacle avoidance
    def _angle(self, v1_x, v1_y, v2_x, v2_y):
        """Calculate angle between two vectors."""
        dot_product = v1_x * v2_x + v1_y * v2_y
        magnitude_1 = np.sqrt(v1_x**2 + v1_y**2)
        magnitude_2 = np.sqrt(v2_x**2 + v2_y**2)
        
        if magnitude_1 * magnitude_2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude_1 * magnitude_2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to avoid numerical errors
        
        angle = np.arccos(cos_angle)
        
        # Determine sign of angle based on cross product
        cross_product = v1_x * v2_y - v1_y * v2_x
        if cross_product < 0:
            angle = -angle
            
        return angle
    
    def _convert_phi(self, phi):
        """Convert angle to phi state (0-2)."""
        if -45 <= phi <= 45:  # Front
            return 0
        elif -135 <= phi < -45 or 45 < phi <= 135:  # Side
            return 1
        else:  # Back
            return 2
    
    def _convert_deltaphi(self, delta_phi):
        """Convert angle change to delta phi state (-2 to 2)."""
        if -15 <= delta_phi <= 15:  # Constant
            return 0
        elif -45 <= delta_phi < -15:  # Less Change
            return -1
        elif 15 < delta_phi <= 45:  # Less Approaching
            return 1
        elif delta_phi < -45:  # Change
            return -2
        else:  # Approaching
            return 2
    
    def _convert_deltad(self, delta_d):
        """Convert distance change to delta d state (-1 to 1)."""
        if -0.5 <= delta_d <= 0.5:  # Constant
            return 0
        elif delta_d < -0.5:  # Change
            return -1
        else:  # Approaching
            return 1