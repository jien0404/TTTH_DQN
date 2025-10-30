import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from controller.Controller import Controller

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        # Feature extraction layers
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Value stream with an additional hidden layer
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Advantage stream with an additional hidden layer
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Combine value and advantage streams
        if len(advantage.shape) == 1:
            q = value + advantage - advantage.mean(dim=0, keepdim=True)
        else:
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

class DQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="dqn_model.pth"):
        # Call parent constructor
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()
    
    def _initialize_algorithm(self):
        """Initialize the DQN algorithm components."""
        print("start....")
        # State and action dimensions
        self.state_dim = 5 * 5 + 1  # 5x5 matrix (25) + distance to goal (1) = 26 dimensions
        self.action_dim = len(self.directions)  # 8 directions
        
        # Initialize networks
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        
        # DQN hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0 if self.is_training else 0.0  # Exploration rate
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.batch_size = 64
        self.memory = deque(maxlen=20000)  # Experience replay buffer
        self.memory.clear()  # Clear old data
        self.target_update_freq = 200  # Update target network every 200 steps
        self.step_count = 0
        
        # Anti-trap mechanism
        self.position_history = {}  # Count of visits to each position
        self.position_memory_size = 100  # Remember only the 100 most recent positions
        self.position_history_list = []  # List of positions in order of visit
        
        # Curiosity mechanism
        self.visit_counts = {}
        self.curiosity_factor = 0.1
        
        # Load model if in testing mode
        if not self.is_training:
            self.load_model()
    
    def make_decision(self, robot, obstacles):
        """Make a decision on the next direction based on current state."""
        # Get current state (5x5 matrix and distance to goal)
        state, distance_to_goal = robot.get_state(obstacles, 32, 32, self.goal)  # Assuming 32x32 grid
        print(state, distance_to_goal)
        # Combine state matrix and distance into a single vector
        state_flat = state.flatten()
        combined_state = np.concatenate([state_flat, [distance_to_goal]])
        state_tensor = torch.FloatTensor(combined_state).to(self.device)

        # Configure network mode
        if self.is_training:
            self.q_network.train()
        else:
            self.q_network.eval()

        # Epsilon-greedy: Choose random action or based on Q-value
        if random.random() < self.epsilon and self.is_training:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                print(q_values)
                action_idx = q_values.argmax().item()

        # Return the corresponding direction
        return self.directions[action_idx]
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer."""
        # state and next_state are tuples (5x5 matrix, distance to goal)
        state_matrix, state_distance = state
        next_state_matrix, next_state_distance = next_state
        # Combine state matrix and distance
        state_flat = state_matrix.flatten()
        next_state_flat = next_state_matrix.flatten()
        state_combined = np.concatenate([state_flat, [state_distance]])
        next_state_combined = np.concatenate([next_state_flat, [next_state_distance]])
        # Convert to tensors
        state_tensor = torch.FloatTensor(state_combined).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state_combined).to(self.device)
        action = torch.LongTensor([action_idx]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        self.memory.append((state_tensor, action, reward, next_state_tensor, done))
    
    def train(self):
        """Perform a training step."""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards).squeeze()  # Force to [batch_size]
        next_states = torch.stack(next_states)
        dones = torch.stack(dones).squeeze()  # Force to [batch_size]

        # Calculate current Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Calculate target Q-values using DDQN
        with torch.no_grad():
            # Use main network to select best actions at next states
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)

            # Use target network to evaluate those actions
            next_q_values = self.target_network(next_states).gather(1, next_actions)

            # Calculate target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.squeeze()

        # Calculate loss and update network
        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Update exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        """Calculate reward for the current state."""
        # Save current position and check for repetition
        position = (robot.grid_x, robot.grid_y)
        
        # Update position history
        if position in self.position_history:
            self.position_history[position] += 1
        else:
            self.position_history[position] = 1
        
        # Add to position list
        self.position_history_list.append(position)
        
        # Keep list size within limit
        if len(self.position_history_list) > self.position_memory_size:
            old_pos = self.position_history_list.pop(0)
            self.position_history[old_pos] -= 1
            if self.position_history[old_pos] <= 0:
                del self.position_history[old_pos]
        
        # Penalty for position repetition
        repetition_penalty = min(-2 * (self.position_history[position] - 1), 0)
        
        # Update visit counts for curiosity
        if position in self.visit_counts:
            self.visit_counts[position] += 1
        else:
            self.visit_counts[position] = 1
        
        # Curiosity reward
        curiosity_reward = self.curiosity_factor / max(1, self.visit_counts[position]**0.5)

        # Base rewards
        if reached_goal:
            return 100 + curiosity_reward  # Large reward for reaching goal
            
        if robot.check_collision(obstacles):
            return -50  # Penalty for collision
        
        # Reward for progress toward goal
        if prev_distance is not None:
            # Larger reward for getting closer to goal
            progress_reward = (prev_distance - distance_to_goal) * 10
            
            # Small penalty for each step to encourage shorter paths
            step_penalty = -0.1
            
            # Larger penalty for moving away from goal
            if distance_to_goal > prev_distance:
                progress_reward -= 5  # Additional penalty for moving away
            total_reward = progress_reward + step_penalty + repetition_penalty + curiosity_reward
            print(total_reward)
            return total_reward
        
        # If no previous distance available
        return -0.1 - (distance_to_goal * 0.05) + repetition_penalty + curiosity_reward
    
    def _save_model_implementation(self):
        """Implement model saving for DQN."""
        torch.save(self.q_network.state_dict(), self.model_path)
    
    def _load_model_implementation(self):
        """Implement model loading for DQN."""
        self.q_network.load_state_dict(torch.load(self.model_path))
        self.q_network.eval()