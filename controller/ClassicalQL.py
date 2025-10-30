import numpy as np
import random
import torch
import os
from controller.Controller import Controller

# Hyperparameters from system old (QLearning)
GAMMA = 0.9  # Discount factor
EPSILON = 0.5  # Initial exploration rate
EPSILON_DECAY = 0.95  # Decay rate for exploration
ALPHA = 0.9  # Learning rate
LEARNING_RATE_DECAY = 1.0  # Learning rate decay

# Reward parameters
COLLISION_DISCOUNT = -0.3
SUCCESS_REWARD = 1.0
MOVE_REWARD = -1.0  # Explicitly define move reward (implicit -1 in old system)

class ClassicalQL(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="models/qtable.pth"):
        """Initialize the Classical Q-Learning controller"""
        # Initialize attributes before super().__init__ to avoid AttributeError
        self.Qtable = {}  # Q-table to store state-action values
        self.policy = {}  # Policy derived from Q-table
        self.episodeDecisions = []  # Store decisions for current episode
        self.sumOfRewards = []  # Track rewards over episodes
        self.averageReward = []  # Track average rewards
        self.epsilon = EPSILON  # Initialize exploration rate
        
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
        # Initialize Q-table and policy
        self._initialize_algorithm()
        
        # Load model if not training
        if not is_training:
            self.load_model()

    def _initialize_algorithm(self):
        """Initialize Q-table and policy as in system old"""
        grid_size = int((512 - 2 * self.env_padding) / self.cell_size)  # Assuming env_size is 512
        for i in range(grid_size):
            for j in range(grid_size):
                # Initialize policy to go toward the goal
                cell_center = (
                    self.env_padding + self.cell_size / 2 + i * self.cell_size,
                    self.env_padding + self.cell_size / 2 + j * self.cell_size
                )
                direction = (self.goal[0] - cell_center[0], self.goal[1] - cell_center[1])
                decision = self._get_initial_decision(direction)
                self.policy[(i, j)] = decision
                
                # Initialize Q-table with zeros
                for direction in self.directions:
                    action_key = self._direction_to_action_key(direction)
                    self.Qtable[(i, j, action_key)] = 0

    def _get_initial_decision(self, direction):
        """Convert direction vector to action decision string (from system old)"""
        decision = ""
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
                
        decision += "_1"
        return decision

    def _direction_to_action_key(self, direction):
        """Convert direction tuple to action key"""
        dx, dy = direction
        if dx == 1 and dy == 0:
            return "right_1"
        elif dx == 1 and dy == 1:
            return "down-right_1"
        elif dx == 0 and dy == 1:
            return "down_1"
        elif dx == -1 and dy == 1:
            return "down-left_1"
        elif dx == -1 and dy == 0:
            return "left_1"
        elif dx == -1 and dy == -1:
            return "up-left_1"
        elif dx == 0 and dy == -1:
            return "up_1"
        elif dx == 1 and dy == -1:
            return "up-right_1"
        else:
            return "right_1"  # Default

    def _action_key_to_direction(self, action_key):
        """Convert action key to direction tuple"""
        if action_key == "right_1":
            return (1, 0)
        elif action_key == "down-right_1":
            return (1, 1)
        elif action_key == "down_1":
            return (0, 1)
        elif action_key == "down-left_1":
            return (-1, 1)
        elif action_key == "left_1":
            return (-1, 0)
        elif action_key == "up-left_1":
            return (-1, -1)
        elif action_key == "up_1":
            return (0, -1)
        elif action_key == "up-right_1":
            return (1, -1)
        else:
            return (1, 0)  # Default

    def _ensure_tuple(self, state):
        """Ensure state is a tuple of integers, handling complex state representations safely"""
        if state is None:
            return (0, 0)  # Default state if None is provided
        
        try:
            # Case where state is a tuple containing (array, float)
            if isinstance(state, tuple) and len(state) == 2:
                if isinstance(state[0], np.ndarray) and isinstance(state[1], (int, float)):
                    # This is the problematic case from the error logs
                    # Extract robot position from the grid representation
                    array_part = state[0]
                    # Find the position of '2' which seems to represent the robot
                    robot_positions = np.where(array_part == 2)
                    if len(robot_positions[0]) > 0 and len(robot_positions[1]) > 0:
                        # Return the position where '2' was found (row, col)
                        return (int(robot_positions[1][0]), int(robot_positions[0][0]))
                    else:
                        # If no robot found, use a default position
                        return (0, 0)
                
            # Handle numpy array
            if isinstance(state, np.ndarray):
                # Handle multi-dimensional arrays
                if state.ndim > 1:
                    # For 2D grid, try to find the position of value 2 (robot)
                    robot_positions = np.where(state == 2)
                    if len(robot_positions[0]) > 0 and len(robot_positions[1]) > 0:
                        # Return the position where '2' was found (row, col)
                        return (int(robot_positions[1][0]), int(robot_positions[0][0]))
                    else:
                        # If no robot found, use first two elements if possible
                        flat_state = state.flatten()
                        if flat_state.size >= 2:
                            return (int(flat_state[0]), int(flat_state[1]))
                        else:
                            return (0, 0)
                else:
                    # For 1D arrays, use the first two elements if available
                    if state.size >= 2:
                        return (int(state[0]), int(state[1]))
                    elif state.size == 1:
                        return (int(state[0]), 0)
                    else:
                        return (0, 0)
            
            # Handle standard tuples/lists
            elif isinstance(state, (list, tuple)):
                if len(state) >= 2:
                    # If the elements are simple numbers, use them directly
                    if all(isinstance(item, (int, float)) for item in state[:2]):
                        return (int(state[0]), int(state[1]))
                    # Handle nested structures
                    else:
                        flattened = []
                        for item in state[:2]:
                            if isinstance(item, (list, tuple, np.ndarray)):
                                # If item is iterable, take first element
                                if len(item) > 0:
                                    if isinstance(item[0], (int, float)):
                                        flattened.append(int(item[0]))
                                    else:
                                        flattened.append(0)
                                else:
                                    flattened.append(0)
                            elif isinstance(item, (int, float)):
                                flattened.append(int(item))
                            else:
                                flattened.append(0)
                        
                        # Ensure we have at least two elements
                        while len(flattened) < 2:
                            flattened.append(0)
                        
                        return (flattened[0], flattened[1])
                else:
                    return (int(state[0]) if len(state) > 0 else 0, 0)
            
            # If we get here, state is likely already a scalar
            else:
                try:
                    return (int(state), int(state))
                except:
                    print(f"Warning: Unable to convert state {state} of type {type(state)} to tuple")
                    return (0, 0)  # Default fallback
        
        except Exception as e:
            print(f"Error in _ensure_tuple: {e}, state: {state}, type: {type(state)}")
            # Return a safe default
            return (0, 0)

    def convertState(self, robot):
        """Convert robot position to grid state"""
        grid_x = int((robot.x - self.env_padding) // self.cell_size)
        grid_y = int((robot.y - self.env_padding) // self.cell_size)
        return (grid_x, grid_y)

    def make_decision(self, robot, obstacles):
        """Make movement decision based on epsilon-greedy (from makeDecision)"""
        self._update_all()
        state = self.convertState(robot)
        state = self._ensure_tuple(state)
        
        if random.random() < self.epsilon:
            direction = random.choice(self.directions)
        else:
            action_key = self.policy.get(state, "right_1")
            direction = self._action_key_to_direction(action_key)
        
        action_key = self._direction_to_action_key(direction)
        self.episodeDecisions.append((state, action_key, MOVE_REWARD))
        return direction

    def store_experience(self, state, action_idx, reward, next_state, done):
        """Update reward for the last decision (emulates setCollision/setSuccess)"""
        try:
            # Convert to tuples for safe dictionary keys
            state = self._ensure_tuple(state)
            next_state = self._ensure_tuple(next_state)
                        
            if self.episodeDecisions:
                last_state, last_action, _ = self.episodeDecisions[-1]
                self.episodeDecisions[-1] = (last_state, last_action, reward)
                self.episodeDecisions.append((next_state, "", 0))
        except Exception as e:
            print(f"Error in store_experience: {e}")
            # Add a default state to prevent further errors
            if self.episodeDecisions:
                last_state, last_action, _ = self.episodeDecisions[-1]
                self.episodeDecisions[-1] = (last_state, last_action, reward)
                self.episodeDecisions.append(((0, 0), "", 0))

    def train(self):
        """Perform training step: update Q-table, policy, and calculate rewards"""
        try:
            self._update_all()
            self._calculate_reward()
            self.episodeDecisions.clear()
            self.update_epsilon()
        except Exception as e:
            print(f"Error in train method: {e}")
            import traceback
            traceback.print_exc()

    def update_epsilon(self):
        """Update exploration rate (from setSuccess)"""
        self.epsilon *= EPSILON_DECAY

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        """Calculate reward (integrates setCollision and setSuccess)"""
        reward = MOVE_REWARD
        if reached_goal:
            reward += SUCCESS_REWARD
        elif done:  # Collision
            reward += COLLISION_DISCOUNT
        return reward

    def updateQtable(self, state, action_key, reward, next_state):
        """Update Q-table using Q-learning update rule (from system old)"""
        try:            
            # Ensure state and next_state are tuples of integers
            state = self._ensure_tuple(state)
            next_state = self._ensure_tuple(next_state)
            
            # Get all possible actions for next state
            next_actions = [self._direction_to_action_key(d) for d in self.directions]
            
            # Find optimal Q-value for next state
            q_values_next = []
            for action in next_actions:
                key = (next_state[0], next_state[1], action)
                q_value = self.Qtable.get(key, 0)
                q_values_next.append(q_value)
                
            max_q_next = max(q_values_next)
            
            # Current Q-value
            old_value = self.Qtable.get((state[0], state[1], action_key), 0)
            
            # Q-learning update rule
            new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * max_q_next)
            
            # Update Q-table
            self.Qtable[(state[0], state[1], action_key)] = new_value
            
            # Update policy for this state
            self.updatePolicy(state)
        
        except Exception as e:
            print(f"Error in updateQtable: {e}")
            import traceback
            traceback.print_exc()

    def updatePolicy(self, state):
        """Update policy based on Q-table (from system old)"""
        try:
            # Ensure state is a tuple of integers
            state = self._ensure_tuple(state)
            
            # Find action with highest Q-value
            best_q_value = float('-inf')
            best_action = None
            
            for direction in self.directions:
                action = self._direction_to_action_key(direction)
                q_value = self.Qtable.get((state[0], state[1], action), 0)
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            if best_action is None:
                best_action = "right_1"  # Default
                
            # Update policy
            self.policy[state] = best_action
            
        except Exception as e:
            print(f"Error in updatePolicy: {e}")
            import traceback
            traceback.print_exc()

    def _update_all(self):
        """Update Q-table and policy from stored decisions (from updateAll)"""
        try:
            if len(self.episodeDecisions) >= 2:
                state, action_key, reward = self.episodeDecisions[-2]
                next_state = self.episodeDecisions[-1][0]
                                
                self.updateQtable(state, action_key, reward, next_state)
        except Exception as e:
            print(f"Error in _update_all: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_reward(self):
        """Calculate sum and average reward for episode (from calculateReward)"""
        try:
            sum_reward = sum(decision[2] for decision in self.episodeDecisions)
            self.sumOfRewards.append(sum_reward)
            self.averageReward.append(sum_reward / max(1, len(self.episodeDecisions)))
        except Exception as e:
            print(f"Error in _calculate_reward: {e}")
            import traceback
            traceback.print_exc()

    def _save_model_implementation(self):
        """Save Q-table and policy to .pth file (adapted for system new, similar to DualQLController)"""
        try:
            # Convert any numpy arrays in keys to tuples
            cleaned_qtable = {}
            for key, value in self.Qtable.items():
                if isinstance(key, tuple):
                    # Process each element of the tuple
                    new_key_parts = []
                    for part in key:
                        if isinstance(part, np.ndarray):
                            if part.size > 0:
                                new_key_parts.append(int(part.item(0)))
                            else:
                                new_key_parts.append(0)
                        else:
                            new_key_parts.append(part)
                    new_key = tuple(new_key_parts)
                else:
                    new_key = key
                cleaned_qtable[new_key] = value
                
            cleaned_policy = {}
            for key, value in self.policy.items():
                if isinstance(key, tuple):
                    # Process each element of the tuple
                    new_key_parts = []
                    for part in key:
                        if isinstance(part, np.ndarray):
                            if part.size > 0:
                                new_key_parts.append(int(part.item(0)))
                            else:
                                new_key_parts.append(0)
                        else:
                            new_key_parts.append(part)
                    new_key = tuple(new_key_parts)
                else:
                    new_key = key
                cleaned_policy[new_key] = value
            
            data = {
                'Qtable': cleaned_qtable,
                'policy': cleaned_policy,
                'epsilon': self.epsilon,
                'sumOfRewards': self.sumOfRewards,
                'averageReward': self.averageReward
            }
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(data, self.model_path)
        except Exception as e:
            print(f"Error in _save_model_implementation: {e}")
            import traceback
            traceback.print_exc()

    def _load_model_implementation(self):
        """Load Q-table and policy from .pth file (adapted for system new, similar to DualQLController)"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file {self.model_path} not found!")
            data = torch.load(self.model_path)
            self.Qtable = data['Qtable']
            self.policy = data['policy']
            self.epsilon = data.get('epsilon', EPSILON)
            self.sumOfRewards = data.get('sumOfRewards', [])
            self.averageReward = data.get('averageReward', [])
        except Exception as e:
            print(f"Error in _load_model_implementation: {e}")
            import traceback
            traceback.print_exc()