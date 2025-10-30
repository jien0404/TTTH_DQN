import numpy as np
import random
import torch
import os
import json
from controller.Controller import Controller

class DFQLController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="models/dfql_model.pth"):
        """
        Initialize the DFQL Controller.
        
        Args:
            goal: The goal position (x, y)
            cell_size: Size of each grid cell
            env_padding: Environment padding
            is_training: Whether we're in training mode
            model_path: Path to save/load model
        """
        # Set DFQL specific hyperparameters before calling super().__init__
        # This ensures they're available when _initialize_algorithm is called
        self.gamma = 0.9  # Discount factor (0.8 to 0.9)
        self.epsilon = 0.5  # Exploration rate
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.alpha = 0.9  # Learning rate (0.2 to 0.9)
        self.learning_rate_decay = 1.0  # Learning rate decay
        self.k_a = 1.5  # Potential field constant
        
        # Reward settings
        self.collision_discount = -10.0
        self.success_reward = 10.0
        
        # Initialize Q-table and tracking variables
        self.q_table = {}
        self.episode_decisions = []
        self.sum_of_rewards = []
        self.average_reward = []
        
        # Now call the parent constructor
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        
    def _initialize_algorithm(self):
        """Initialize DFQL algorithm and its components."""
        # Calculate env size from cell size
        grid_width = int((self.goal[0] * 2) // self.cell_size)
        grid_height = int((self.goal[1] * 2) // self.cell_size)
        self.env_size = max(grid_width, grid_height) * self.cell_size
        
        # Calculate max potential energy
        relative_goal = (self.goal[0] - self.env_padding, self.goal[1] - self.env_padding)
        self.u_max = 0.5 * self.k_a * (
            max(relative_goal[0], self.env_size - relative_goal[0]) ** 2 + 
            max(relative_goal[1], self.env_size - relative_goal[1]) ** 2
        )
        
        # Initialize Q-table
        if os.path.exists(self.model_path) and not self.is_training:
            self.load_model()
        else:
            self._initialize_q_table()
    
    def _initialize_q_table(self):
        """Initialize Q-table with potential field values."""
        # Calculate grid dimensions
        grid_width = int(self.env_size // self.cell_size)
        grid_height = int(self.env_size // self.cell_size)
        
        # Get goal grid position
        goal_grid_x = int((self.goal[0] - self.env_padding) // self.cell_size)
        goal_grid_y = int((self.goal[1] - self.env_padding) // self.cell_size)
        
        # Initialize Q-table for all grid positions and actions
        for i in range(grid_width):
            for j in range(grid_height):
                # Calculate distance to goal
                distance = ((i - goal_grid_x) ** 2 + (j - goal_grid_y) ** 2)
                
                # Calculate potential energy
                u_t = 0.5 * self.k_a * distance
                
                # Initial Q-value based on potential field
                init_value = abs((self.u_max - u_t) / self.u_max)
                
                # Initialize Q-values for all actions
                for action_idx in range(len(self.directions)):
                    state_key = self._get_state_key((i, j), action_idx)
                    self.q_table[state_key] = init_value
    
    def _get_state_key(self, state, action_idx=None):
        """Create a hashable key for the state-action pair."""
        if action_idx is not None:
            return (state[0], state[1], action_idx)
        return (state[0], state[1])
    
    def make_decision(self, robot, obstacles):
        """
        Make a decision about the next movement direction.
        
        Args:
            robot: The robot object
            obstacles: List of obstacles
            
        Returns:
            Direction as (x, y) tuple
        """
        # Get current state
        state = (robot.grid_x, robot.grid_y)
        
        # Epsilon-greedy policy
        if self.is_training and random.random() < self.epsilon:
            # Exploration: choose random action
            action_idx = random.randint(0, len(self.directions) - 1)
        else:
            # Exploitation: choose best action
            action_idx = self._get_best_action(state)
        
        return self.directions[action_idx]
    
    def _get_best_action(self, state):
        """Get the action with the highest Q-value for the given state."""
        q_values = []
        for action_idx in range(len(self.directions)):
            state_key = self._get_state_key(state, action_idx)
            if state_key in self.q_table:
                q_values.append(self.q_table[state_key])
            else:
                # If state-action pair not in Q-table, initialize it
                self.q_table[state_key] = 0.0
                q_values.append(0.0)
        
        return np.argmax(q_values)
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """
        Store experience in memory for training.
        
        Args:
            state: Current state (grid matrix, distance)
            action_idx: Index of the action taken
            reward: Reward received
            next_state: Next state (grid matrix, distance)
            done: Whether the episode is done
        """
        if not self.is_training:
            return
        
        # Extract grid positions from state tuple (state_matrix, distance)
        grid_state = (state[0].shape[0], state[0].shape[1])  # Use matrix shape for now
        next_grid_state = (next_state[0].shape[0], next_state[0].shape[1])
        
        # Store the experience for later updates
        self.episode_decisions.append((grid_state, action_idx, reward, next_grid_state, done))
    
    def train(self):
        """Perform a training step using the latest experience."""
        if not self.is_training or len(self.episode_decisions) == 0:
            return
        
        # Get the most recent experience
        state, action_idx, reward, next_state, done = self.episode_decisions[-1]
        
        # Update Q-table
        self._update_q_value(state, action_idx, reward, next_state, done)
    
    def _update_q_value(self, state, action_idx, reward, next_state, done):
        """Update Q-value using the DFQL algorithm."""
        state_key = self._get_state_key(state, action_idx)
        
        if done:
            # If episode is done, no future rewards
            target = reward
        else:
            # Get best next action's Q-value
            best_next_action = self._get_best_action(next_state)
            next_state_key = self._get_state_key(next_state, best_next_action)
            
            # If next state-action not in Q-table, initialize it
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = 0.0
            
            # Q-learning update target
            target = reward + self.gamma * self.q_table[next_state_key]
        
        # If current state-action not in Q-table, initialize it
        if state_key not in self.q_table:
            self.q_table[state_key] = 0.0
        
        # Update Q-value
        self.q_table[state_key] = (1 - self.alpha) * self.q_table[state_key] + self.alpha * target
    
    def update_epsilon(self):
        """Update exploration rate."""
        if self.is_training:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        """
        Calculate reward for the current state.
        
        Args:
            robot: The robot object
            obstacles: List of obstacles
            done: Whether the episode is done
            reached_goal: Whether the goal was reached
            distance_to_goal: Current distance to goal
            prev_distance: Previous distance to goal
            
        Returns:
            The calculated reward
        """
        if reached_goal:
            return self.success_reward
        
        if done:  # Collision or out of bounds
            return self.collision_discount
        
        # Calculate directional reward component
        if prev_distance is not None and abs(prev_distance - distance_to_goal) > 1e-6:
            r_d = (prev_distance - distance_to_goal) / abs(prev_distance - distance_to_goal)
        else:
            r_d = 0
        
        # Calculate movement type reward component
        # If moving diagonally, reward is reduced
        direction = robot.direction
        if direction[0] != 0 and direction[1] != 0:
            r_s = 1 / np.sqrt(2)  # Diagonal movement
        else:
            r_s = 1  # Cardinal movement
        
        # Combined reward
        reward = r_s * (1 + r_d)
        
        return reward
    
    def _save_model_implementation(self):
        """Save Q-table to disk."""
        # Convert Q-table keys to strings for JSON serialization
        serializable_q_table = {}
        for key, value in self.q_table.items():
            serializable_q_table[str(key)] = float(value)
        
        # Save as JSON
        with open(self.model_path, 'w') as f:
            json.dump({
                'q_table': serializable_q_table,
                'epsilon': self.epsilon,
                'alpha': self.alpha
            }, f)
    
    def _load_model_implementation(self):
        """Load Q-table from disk."""
        with open(self.model_path, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples
        q_table = {}
        for key_str, value in data['q_table'].items():
            # Parse key string to tuple: "(x, y, action)" -> (x, y, action)
            key = eval(key_str)
            q_table[key] = value
        
        self.q_table = q_table
        self.epsilon = data.get('epsilon', self.epsilon)
        self.alpha = data.get('alpha', self.alpha)