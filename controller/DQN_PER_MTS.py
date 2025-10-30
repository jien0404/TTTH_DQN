import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from controller.Controller import Controller

class SumTree:
    """Sum Tree for efficient sampling in PER"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.pending_idx + self.capacity - 1
        self.data[self.pending_idx] = data
        self.update(idx, p)
        self.pending_idx += 1
        if self.pending_idx >= self.capacity:
            self.pending_idx = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayMemory:
    """Prioritized Experience Replay Memory"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        self.max_priority = 1.0

    def add(self, experience):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if data is not None:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

        if len(batch) == 0:
            return None, None, None

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, batch_idxs, batch_priorities):
        for idx, priority in zip(batch_idxs, batch_priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        if len(advantage.shape) == 1:
            q = value + advantage - advantage.mean(dim=0, keepdim=True)
        else:
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

class DQNPMController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="dqn_model.pth"):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()
        if not self.is_training:
            self.load_model()

    def _initialize_algorithm(self):
        print("Initializing DQNController with PER and Multi-Step DDQN...")
        self.state_dim = 5 * 5 + 1  # 5x5 matrix + distance to goal
        self.action_dim = len(self.directions)  # 8 directions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)

        # DQN hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0 if self.is_training else 0.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.batch_size = 64
        self.target_update_freq = 200
        self.step_count = 0

        # Prioritized Experience Replay
        self.memory = PrioritizedReplayMemory(
            capacity=20000, 
            alpha=0.6, 
            beta=0.4, 
            beta_increment=0.001
        )

        # Multi-step learning - Enhanced
        self.n_steps = 3  # Number of steps for multi-step learning
        self.multi_step_buffer = deque(maxlen=self.n_steps)
        self.gamma_n = self.gamma ** self.n_steps

        # Anti-trap and curiosity
        self.position_history = {}
        self.position_memory_size = 100
        self.position_history_list = []
        self.visit_counts = {}
        self.curiosity_factor = 0.1

        # if not self.is_training:
        #     self.load_model()

    def make_decision(self, robot, obstacles):
        state, distance_to_goal = robot.get_state(obstacles, 32, 32, self.goal)
        state_flat = state.flatten()
        combined_state = np.concatenate([state_flat, [distance_to_goal]])
        state_tensor = torch.FloatTensor(combined_state).to(self.device)

        if self.is_training:
            self.q_network.train()
        else:
            self.q_network.eval()

        if random.random() < self.epsilon and self.is_training:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()

        return self.directions[action_idx]

    def store_experience(self, state, action_idx, reward, next_state, done):
        state_matrix, state_distance = state
        next_state_matrix, next_state_distance = next_state
        state_flat = state_matrix.flatten()
        next_state_flat = next_state_matrix.flatten()
        state_combined = np.concatenate([state_flat, [state_distance]])
        next_state_combined = np.concatenate([next_state_flat, [next_state_distance]])

        # Store in multi-step buffer
        self.multi_step_buffer.append((state_combined, action_idx, reward, next_state_combined, done))

        # Process multi-step experience when buffer is full or episode ends
        if len(self.multi_step_buffer) == self.n_steps or done:
            # Calculate n-step return with proper gamma discounting
            n_step_reward = 0
            gamma_power = 1
            
            for i, (s, a, r, ns, d) in enumerate(self.multi_step_buffer):
                n_step_reward += gamma_power * r
                gamma_power *= self.gamma
                if d and i < len(self.multi_step_buffer) - 1:
                    # Episode ended early, adjust gamma
                    self.gamma_n = self.gamma ** (i + 1)
                    break
            else:
                # Full n-step sequence
                self.gamma_n = self.gamma ** self.n_steps
            
            # Get first and last states for n-step transition
            n_step_state = self.multi_step_buffer[0][0]
            n_step_action = self.multi_step_buffer[0][1]
            n_step_next_state = self.multi_step_buffer[-1][3]
            n_step_done = self.multi_step_buffer[-1][4]

            # Store in prioritized replay memory
            experience = (n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)
            self.memory.add(experience)

            # Clear buffer if episode ended
            if done:
                self.multi_step_buffer.clear()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
    
        # Sample from prioritized replay memory
        batch, batch_idxs, importance_weights = self.memory.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        importance_weights = torch.FloatTensor(importance_weights).to(self.device)
    
        # Calculate current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    
        # Calculate target Q-values using Double DQN with multi-step returns
        with torch.no_grad():
            # Double DQN: use main network to select actions, target network to evaluate
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma_n * next_q_values
    
        # Calculate TD errors for priority updates
        td_errors = torch.abs(current_q_values - target_q_values)
        
        # Calculate weighted loss using importance sampling weights
        loss = (importance_weights * (current_q_values - target_q_values).pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities in replay memory
        priorities = td_errors.detach().cpu().numpy()
        self.memory.update_priorities(batch_idxs, priorities)
    
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        position = (robot.grid_x, robot.grid_y)
        
        # Update position history for anti-trap mechanism
        if position in self.position_history:
            self.position_history[position] += 1
        else:
            self.position_history[position] = 1
        
        self.position_history_list.append(position)
        
        # Maintain position history size
        if len(self.position_history_list) > self.position_memory_size:
            old_pos = self.position_history_list.pop(0)
            self.position_history[old_pos] -= 1
            if self.position_history[old_pos] <= 0:
                del self.position_history[old_pos]
        
        # Calculate repetition penalty
        repetition_penalty = min(-2 * (self.position_history[position] - 1), 0)
        
        # Update visit counts for curiosity
        if position in self.visit_counts:
            self.visit_counts[position] += 1
        else:
            self.visit_counts[position] = 1
        
        # Calculate curiosity reward
        curiosity_reward = self.curiosity_factor / max(1, self.visit_counts[position]**0.5)

        # Goal reached - highest reward
        if reached_goal:
            return 100 + curiosity_reward
            
        # Collision penalty
        if robot.check_collision(obstacles):
            return -50
        
        # Progress-based reward
        if prev_distance is not None:
            progress_reward = (prev_distance - distance_to_goal) * 10
            step_penalty = -0.1
            if distance_to_goal > prev_distance:
                progress_reward -= 5
            return progress_reward + step_penalty + repetition_penalty + curiosity_reward
        
        # Default reward
        return -0.1 - (distance_to_goal * 0.05) + repetition_penalty + curiosity_reward

    def save_model(self):
        torch.save(self.q_network.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.q_network.load_state_dict(torch.load(self.model_path))
            self.q_network.eval()
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file {self.model_path} not found!")