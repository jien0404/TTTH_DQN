# controller/AdvancedDQNController.py

import numpy as np
import torch
import pygame
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import os
from controller.Controller import Controller
from utils.gpu_utils import find_free_gpu

# Define Transition at module level (CRITICAL for pickling)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# --- Prioritized Replay Buffer với Anti-Catastrophic Forgetting ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, good_capacity=5000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

        # CRITICAL: Good memory buffer để tránh catastrophic forgetting
        self.good_buffer = []
        self.good_capacity = good_capacity
        self.good_pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        transition = Transition(state, action, reward, next_state, done)

        # Store in main buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

        # CRITICAL: Store good experiences separately
        if reward > 50:  # Goal reached or significant progress
            if len(self.good_buffer) < self.good_capacity:
                self.good_buffer.append(transition)
            else:
                self.good_buffer[self.good_pos] = transition
            self.good_pos = (self.good_pos + 1) % self.good_capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None, None, None

        # CRITICAL: Mix 75% from regular buffer + 25% from good buffer
        good_sample_size = min(len(self.good_buffer), max(1, int(batch_size * 0.25)))
        regular_sample_size = batch_size - good_sample_size

        # Sample from regular buffer with prioritization
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        regular_indices = np.random.choice(len(self.buffer), regular_sample_size, p=probs, replace=False)
        regular_samples = [self.buffer[idx] for idx in regular_indices]

        # Sample from good buffer uniformly
        good_samples = []
        if len(self.good_buffer) > 0:
            good_indices = np.random.choice(len(self.good_buffer), good_sample_size, replace=False)
            good_samples = [self.good_buffer[idx] for idx in good_indices]

        # Combine samples
        all_samples = regular_samples + good_samples

        # Calculate importance sampling weights (only for regular samples)
        total = len(self.buffer)
        weights_regular = (total * probs[regular_indices]) ** (-beta)
        weights_regular /= weights_regular.max()

        # Good samples get weight 1.0
        weights_good = np.ones(len(good_samples), dtype=np.float32)
        weights = np.concatenate([weights_regular, weights_good])

        batch = Transition(*zip(*all_samples))
        return batch, regular_indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

    def good_buffer_size(self):
        return len(self.good_buffer)


# --- Hybrid A* for Path Planning ---
import heapq


class HybridAStar:
    def __init__(self, grid_width, grid_height):
        self.grid_w = grid_width
        self.grid_h = grid_height
        self.cache = {}

    def heuristic(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def neighbors(self, x, y):
        # 8 directions with costs
        dirs = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        for dx, dy, cost in dirs:
            yield x + dx, y + dy, cost

    def find_path(self, start, goal, obstacle_set):
        # Cache check
        cache_key = (start, goal, frozenset(obstacle_set))
        if cache_key in self.cache:
            return self.cache[cache_key]

        sx, sy = start
        gx, gy = goal

        open_set = []
        heapq.heappush(open_set, (0, sx, sy))
        came_from = {}
        g_score = {start: 0}

        max_iterations = 1000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, x, y = heapq.heappop(open_set)

            if (x, y) == (gx, gy):
                # Reconstruct path
                path = []
                current = (x, y)
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                self.cache[cache_key] = path
                return path

            for nx, ny, move_cost in self.neighbors(x, y):
                if (nx, ny) in obstacle_set:
                    continue
                if nx < 0 or ny < 0 or nx >= self.grid_w or ny >= self.grid_h:
                    continue

                ng = g_score[(x, y)] + move_cost
                if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = ng
                    priority = ng + self.heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_set, (priority, nx, ny))
                    came_from[(nx, ny)] = (x, y)

        self.cache[cache_key] = []
        return []


# --- Noisy Linear Layer for exploration ---
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


# --- Enhanced Dueling DQN with Noisy Layers ---
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Value stream with noisy layers
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)
        )

        # Advantage stream with noisy layers
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Dueling aggregation
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class AdvancedDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="advanced_dqn_model.pth",
                 grid_width=32, grid_height=32):
        # Set default grid size if not provided
        self.grid_width = grid_width
        self.grid_height = grid_height
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("🚀 Initializing AdvancedDQNController with Anti-Catastrophic Forgetting...")

        self.state_dim = 5 * 5 + 1
        self.action_dim = len(self.directions)

        # Auto select free GPU
        gpu_id = find_free_gpu()
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")
        print(f"📍 Using device: {self.device}")

        # Networks
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1000, T_mult=2)

        # Training hyperparameters
        self.gamma = 0.995
        self.epsilon = 0.5 if self.is_training else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9997
        self.batch_size = 128

        # Hybrid A* weight (how much to trust A* vs DQN)
        self.hybrid_weight = 0.8 if self.is_training else 0.0  # High trust in A* initially
        self.hybrid_decay = 0.9998  # Gradually trust DQN more
        self.hybrid_min = 0.3

        # CRITICAL: Enhanced replay buffer with good memory
        self.memory = PrioritizedReplayBuffer(capacity=50000, alpha=0.6, good_capacity=10000)
        self.beta_start = 0.4
        self.beta_frames = 100000

        # Target network update
        self.target_update_freq = 500
        self.frame_idx = 0

        # Anti-loop mechanisms (ENHANCED)
        self.position_history = {}
        self.position_memory_size = 150
        self.position_history_list = []
        self.visit_counts = {}
        self.curiosity_factor = 0.1
        self.recent_positions = deque(maxlen=10)  # Track last 10 positions

        # Hybrid A* planner
        self.astar = HybridAStar(self.grid_width, self.grid_height)

        # Performance tracking
        self.recent_rewards = deque(maxlen=100)
        self.steps_since_goal = 0
        self.episode_steps = 0

        if not self.is_training:
            self.load_model()

    def reset_episode(self):
        """Call at the start of each episode"""
        self.position_history.clear()
        self.position_history_list.clear()
        self.recent_positions.clear()
        self.steps_since_goal = 0
        self.episode_steps = 0
        self.astar.cache.clear()  # Clear A* cache for new episode

    def make_decision(self, robot, obstacles):
        state, distance_to_goal = robot.get_state(obstacles, self.grid_width, self.grid_height, self.goal)
        state_flat = state.flatten()
        combined_state = np.concatenate([state_flat, [distance_to_goal]])
        state_tensor = torch.FloatTensor(combined_state).unsqueeze(0).to(self.device)

        # Get current position
        robot_grid = (int(robot.x // self.cell_size), int(robot.y // self.cell_size))
        goal_grid = (int(self.goal[0] // self.cell_size), int(self.goal[1] // self.cell_size))
        self.recent_positions.append(robot_grid)

        # Build obstacle set for A*
        obstacle_set = set()
        for obs in obstacles:
            if obs.static:
                obs_grid_x = int(obs.x // self.cell_size)
                obs_grid_y = int(obs.y // self.cell_size)
                # Add obstacle cells with some padding
                for dx in range(-1, int(obs.width // self.cell_size) + 2):
                    for dy in range(-1, int(obs.height // self.cell_size) + 2):
                        obstacle_set.add((obs_grid_x + dx, obs_grid_y + dy))

        # Get A* path suggestion
        astar_path = self.astar.find_path(robot_grid, goal_grid, obstacle_set)
        astar_action = None

        if len(astar_path) > 0:
            next_pos = astar_path[0]
            dx = next_pos[0] - robot_grid[0]
            dy = next_pos[1] - robot_grid[1]

            # Find matching direction
            for i, direction in enumerate(self.directions):
                if direction == (dx, dy):
                    astar_action = i
                    break

        # Action masking
        valid_mask = self._get_valid_action_mask(robot, obstacles)

        # Penalize recently visited positions in action selection
        for i, direction in enumerate(self.directions):
            next_pos = (robot_grid[0] + direction[0], robot_grid[1] + direction[1])
            # Count how many times this position appears in recent history
            recent_count = sum(1 for p in self.recent_positions if p == next_pos)
            if recent_count >= 3:  # If visited 3+ times recently, discourage
                valid_mask[i] = False

        # Decision making
        if self.is_training:
            self.q_network.train()
            self.q_network.reset_noise()

            # Hybrid decision: A* + DQN
            if random.random() < self.epsilon:
                # Epsilon exploration
                valid_indices = [i for i, v in enumerate(valid_mask) if v]
                if valid_indices:
                    action_idx = random.choice(valid_indices)
                else:
                    action_idx = random.randint(0, self.action_dim - 1)
            else:
                # Exploit: blend A* and DQN
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    # Apply mask
                    invalid_actions = torch.tensor([not v for v in valid_mask], dtype=torch.bool).to(self.device)
                    q_values[0][invalid_actions] = -1e8
                    dqn_action = q_values.argmax(dim=1).item()

                # Use A* suggestion with probability hybrid_weight
                if astar_action is not None and valid_mask[astar_action] and random.random() < self.hybrid_weight:
                    action_idx = astar_action
                else:
                    action_idx = dqn_action
        else:
            # Inference mode
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                invalid_actions = torch.tensor([not v for v in valid_mask], dtype=torch.bool).to(self.device)
                q_values[0][invalid_actions] = -1e8
                action_idx = q_values.argmax(dim=1).item()

        self.episode_steps += 1
        return self.directions[action_idx]

    def _get_valid_action_mask(self, robot, obstacles):
        """Prevent invalid actions (collision with walls/obstacles)"""
        mask = [True] * self.action_dim
        for i, direction in enumerate(self.directions):
            next_x = robot.x + direction[0] * self.cell_size
            next_y = robot.y + direction[1] * self.cell_size

            # Check wall collision
            if not (self.env_padding <= next_x < self.env_padding + self.grid_width * self.cell_size and
                    self.env_padding <= next_y < self.env_padding + self.grid_height * self.cell_size):
                mask[i] = False
                continue

            # Check static obstacle collision
            for obs in obstacles:
                if obs.static:
                    obstacle_rect = pygame.Rect(obs.x, obs.y, obs.width, obs.height)
                    robot_next_rect = pygame.Rect(next_x - robot.radius, next_y - robot.radius, robot.radius * 2,
                                                  robot.radius * 2)
                    if obstacle_rect.colliderect(robot_next_rect):
                        mask[i] = False
                        break

        # If all actions are invalid, allow all (safety fallback)
        if not any(mask):
            mask = [True] * self.action_dim

        return mask

    def store_experience(self, state, action_idx, reward, next_state, done):
        state_matrix, state_distance = state
        next_matrix, next_distance = next_state
        state_flat = state_matrix.flatten()
        next_flat = next_matrix.flatten()
        state_combined = np.concatenate([state_flat, [state_distance]])
        next_combined = np.concatenate([next_flat, [next_distance]])

        self.memory.push(state_combined, action_idx, reward, next_combined, done)
        self.recent_rewards.append(reward)

    def train(self):
        """Train the network with mixed sampling from regular and good buffers"""
        self.frame_idx += 1

        if len(self.memory) < self.batch_size:
            return

        # Anneal beta for importance sampling
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

        batch, indices, weights = self.memory.sample(self.batch_size, beta)

        if batch is None:
            return

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.q_network(states).gather(1, actions)

        # Double DQN: select action with online network, evaluate with target network
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values

        # Weighted Huber loss
        loss = nn.SmoothL1Loss(reduction='none')(q_values, target_q_values)
        loss = (loss * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities (only for regular samples, not good samples)
        td_errors = (target_q_values - q_values).abs().detach().cpu().numpy().squeeze()
        # Only update priorities for regular buffer samples
        if len(indices) > 0:
            self.memory.update_priorities(indices, td_errors[:len(indices)] + 1e-6)

        # Hard update target network
        if self.frame_idx % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"🎯 Target network updated at frame {self.frame_idx} | Good buffer: {self.memory.good_buffer_size()}")

    def update_epsilon(self):
        """Decay epsilon and hybrid_weight after each episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

        # Gradually reduce trust in A*, trust DQN more over time
        if self.hybrid_weight > self.hybrid_min:
            self.hybrid_weight *= self.hybrid_decay
            self.hybrid_weight = max(self.hybrid_min, self.hybrid_weight)

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        """Enhanced reward function with multiple shaping components"""
        position = (robot.grid_x, robot.grid_y)

        # Track position history
        if position in self.position_history:
            self.position_history[position] += 1
        else:
            self.position_history[position] = 1

        self.position_history_list.append(position)
        if len(self.position_history_list) > self.position_memory_size:
            old = self.position_history_list.pop(0)
            if old in self.position_history:
                self.position_history[old] -= 1
                if self.position_history[old] <= 0:
                    del self.position_history[old]

        # ENHANCED: Repetition penalty (very strong to break loops)
        repetition_count = self.position_history.get(position, 1) - 1
        repetition_penalty = -10.0 * min(repetition_count, 5) if repetition_count > 1 else 0

        # ENHANCED: Stagnation penalty (prevent tight loops)
        stagnation_penalty = 0
        history_length = 20
        if len(self.position_history_list) > history_length:
            recent_history = self.position_history_list[-history_length:]
            num_unique_positions = len(set(recent_history))
            # If stuck in small area (< 6 unique positions in 20 steps)
            if num_unique_positions <= 5:
                stagnation_penalty = -15.0
            elif num_unique_positions <= 8:
                stagnation_penalty = -8.0

        # Back-and-forth penalty
        backforth_penalty = 0
        if len(self.position_history_list) >= 4:
            last_4 = self.position_history_list[-4:]
            # Check if oscillating between 2 positions
            if len(set(last_4)) <= 2:
                backforth_penalty = -12.0

        # Curiosity reward
        if position in self.visit_counts:
            self.visit_counts[position] += 1
        else:
            self.visit_counts[position] = 1
        curiosity_reward = self.curiosity_factor / (self.visit_counts[position] ** 0.5)

        # Goal reached - BIG reward with time bonus
        if reached_goal:
            time_bonus = max(0, 200 - self.episode_steps * 0.5)
            return 300.0 + time_bonus + curiosity_reward

        # Collision penalty
        if robot.check_collision(obstacles):
            return -150.0

        # Progress reward (ENHANCED)
        if prev_distance is not None:
            progress = prev_distance - distance_to_goal
            # Strong reward for moving toward goal, strong penalty for moving away
            progress_reward = progress * 25.0 if progress > 0 else progress * 20.0

            # Distance-based shaping (encourage being closer to goal)
            distance_reward = -distance_to_goal * 0.03

            # Time penalty (encourage faster completion)
            step_penalty = -0.2

            self.steps_since_goal += 1

            return (progress_reward + distance_reward + step_penalty +
                    repetition_penalty + curiosity_reward + stagnation_penalty + backforth_penalty)

        return -0.1 + curiosity_reward + repetition_penalty + stagnation_penalty

    def save_model(self):
        """Save complete checkpoint"""
        # Convert good_buffer to serializable format
        good_buffer_data = []
        if hasattr(self.memory, 'good_buffer'):
            for trans in self.memory.good_buffer:
                good_buffer_data.append({
                    'state': trans.state,
                    'action': trans.action,
                    'reward': trans.reward,
                    'next_state': trans.next_state,
                    'done': trans.done
                })

        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'hybrid_weight': self.hybrid_weight,
            'frame_idx': self.frame_idx,
            'visit_counts': self.visit_counts,
            'good_buffer_data': good_buffer_data  # Save as dict instead of namedtuple
        }
        torch.save(checkpoint, self.model_path)
        print(
            f"💾 Model saved | Good buffer: {len(good_buffer_data)} | ε={self.epsilon:.4f} | hybrid={self.hybrid_weight:.4f}")

    def load_model(self):
        """Load complete checkpoint"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])

            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'hybrid_weight' in checkpoint:
                self.hybrid_weight = checkpoint['hybrid_weight']
            if 'frame_idx' in checkpoint:
                self.frame_idx = checkpoint['frame_idx']
            if 'visit_counts' in checkpoint:
                self.visit_counts = checkpoint['visit_counts']

            # Reconstruct good_buffer from saved data
            if 'good_buffer_data' in checkpoint and hasattr(self.memory, 'good_buffer'):
                self.memory.good_buffer = []
                for data in checkpoint['good_buffer_data']:
                    trans = Transition(
                        state=data['state'],
                        action=data['action'],
                        reward=data['reward'],
                        next_state=data['next_state'],
                        done=data['done']
                    )
                    self.memory.good_buffer.append(trans)
                print(f"✅ Loaded {len(self.memory.good_buffer)} good experiences")

            self.q_network.eval()
            print(f"✅ Model loaded | ε={self.epsilon:.4f} | hybrid={self.hybrid_weight:.4f}")
        else:
            print(f"⚠️ Model file not found: {self.model_path}")

    def _save_model_implementation(self):
        """Override base class method"""
        self.save_model()

    def _load_model_implementation(self):
        """Override base class method"""
        self.load_model()