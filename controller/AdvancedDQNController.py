import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from controller.Controller import Controller
import heapq


# ---------------- Dueling DQN with Noisy Layers ------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
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
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ---------------- Improved Hybrid A* ------------------
class HybridAStar:
    def __init__(self, grid_size=(32, 32)):
        self.grid_w, self.grid_h = grid_size
        self.cache = {}
        self.cache_hits = 0

    def heuristic(self, x1, y1, x2, y2):
        # Euclidean distance is better than Manhattan for diagonal movement
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def neighbors(self, x, y):
        # Prioritize straight moves over diagonal
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for i, (dx, dy) in enumerate(dirs):
            cost = 1.414 if i >= 4 else 1.0  # diagonal costs more
            yield x + dx, y + dy, cost

    def find_path(self, start, goal, obstacle_set):
        # Cache check
        cache_key = (start, goal, frozenset(obstacle_set))
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        sx, sy = start
        gx, gy = goal
        open_set = []
        heapq.heappush(open_set, (0, sx, sy))
        came_from = {}
        g_score = {start: 0}

        max_iterations = 500  # Prevent infinite loops
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, x, y = heapq.heappop(open_set)

            if (x, y) == (gx, gy):
                self.cache[cache_key] = came_from
                return came_from

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

        self.cache[cache_key] = {}
        return {}


# ---------------- Advanced DQN Controller ------------------
class AdvancedDQNController(Controller):
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="dqn_model.pth"):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)
        self._initialize_algorithm()

    def _initialize_algorithm(self):
        print("Initializing Improved AdvancedDQNController...")

        # State & action
        self.state_dim = 5 * 5 + 1
        self.action_dim = len(self.directions)

        # Networks
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer with better hyperparams
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=0.0001, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1000, T_mult=2)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.memory = deque(maxlen=50000)
        self.priorities = deque(maxlen=50000)
        self.batch_size = 128
        self.gamma = 0.995  # Slightly higher for longer horizon
        self.step_count = 0
        self.target_update_freq = 500
        self.tau = 0.001  # Softer update

        # Exploration - start lower for Noisy Networks
        self.epsilon = 0.5 if self.is_training else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.hybrid_weight = 0.7  # Trust A* more initially
        self.hybrid_decay = 0.9998  # Slowly reduce over time

        # Anti-loop + curiosity
        self.position_history = {}
        self.visit_counts = {}
        self.episode_positions = []

        # Hybrid A*
        self.hybrid = HybridAStar(grid_size=(32, 32))

        # Performance tracking
        self.recent_rewards = deque(maxlen=100)
        self.steps_since_goal = 0

        # Load model if not training
        if not self.is_training:
            self.load_model()

    def reset_episode(self):
        """Call this at the start of each episode"""
        self.position_history.clear()
        self.episode_positions = []
        self.steps_since_goal = 0

    def make_decision(self, robot, obstacles):
        state, distance_to_goal = robot.get_state(obstacles, 32, 32, self.goal)
        state_flat = state.flatten()
        combined_state = np.concatenate([state_flat, [distance_to_goal]])
        state_tensor = torch.FloatTensor(combined_state).unsqueeze(0).to(self.device)

        if self.is_training:
            self.q_network.train()
            self.q_network.reset_noise()
        else:
            self.q_network.eval()

        # Track position
        robot_grid = (int(robot.x // self.cell_size), int(robot.y // self.cell_size))
        self.episode_positions.append(robot_grid)

        # Hybrid A* guidance with caching
        obstacle_set = set((int(o.x // self.cell_size), int(o.y // self.cell_size)) for o in obstacles)
        goal_grid = (int(self.goal[0] // self.cell_size), int(self.goal[1] // self.cell_size))

        path_map = self.hybrid.find_path(robot_grid, goal_grid, obstacle_set)

        hybrid_suggest = None
        if path_map and goal_grid in path_map or any(goal_grid == k for k in path_map.keys()):
            cur = goal_grid
            # Reconstruct next step
            path = [cur]
            while cur in path_map and path_map[cur] != robot_grid:
                cur = path_map[cur]
                path.append(cur)
                if len(path) > 100:  # Safety check
                    break

            if len(path) > 1:
                next_step = path[-2]
                dx = next_step[0] - robot_grid[0]
                dy = next_step[1] - robot_grid[1]

                for i, d in enumerate(self.directions):
                    if d == (dx, dy):
                        hybrid_suggest = i
                        break

        # Decision making
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            dqn_choice = q_values.argmax().item()

        # Epsilon-greedy with hybrid guidance
        if random.random() < self.epsilon and self.is_training:
            if hybrid_suggest is not None and random.random() < self.hybrid_weight:
                action_idx = hybrid_suggest
            else:
                action_idx = random.randint(0, self.action_dim - 1)
        else:
            if hybrid_suggest is not None and random.random() < self.hybrid_weight:
                action_idx = hybrid_suggest
            else:
                action_idx = dqn_choice

        return self.directions[action_idx]

    def store_experience(self, state, action_idx, reward, next_state, done):
        state_matrix, state_distance = state
        next_matrix, next_distance = next_state
        state_flat = state_matrix.flatten()
        next_flat = next_matrix.flatten()
        state_combined = np.concatenate([state_flat, [state_distance]])
        next_combined = np.concatenate([next_flat, [next_distance]])

        state_tensor = torch.FloatTensor(state_combined).to(self.device)
        next_tensor = torch.FloatTensor(next_combined).to(self.device)
        action_tensor = torch.LongTensor([action_idx]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)

        self.memory.append((state_tensor, action_tensor, reward_tensor, next_tensor, done_tensor))

        # Initialize priority
        max_priority = max([float(p) for p in self.priorities], default=1.0)
        self.priorities.append(float(max_priority))

        self.recent_rewards.append(reward)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Prioritized sampling
        clean_priorities = np.array([float(p) for p in self.priorities], dtype=np.float32)
        clean_priorities = np.clip(clean_priorities, 1e-6, None)
        probs = clean_priorities ** 0.6
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs, replace=False)
        batch = [self.memory[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards).squeeze()
        next_states = torch.stack(next_states)
        dones = torch.stack(dones).squeeze()

        # Double DQN
        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values.squeeze()

        loss = self.loss_fn(q_values, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        td_errors = (q_values - target_q.unsqueeze(1)).detach().cpu().numpy()
        for i, idx in enumerate(indices):
            err = float(np.abs(td_errors[i]).item())
            self.priorities[idx] = err + 1e-4

        # Soft update target
        if self.step_count % self.target_update_freq == 0:
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        self.step_count += 1

    def update_epsilon(self):
        """Call this after each episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

        if self.hybrid_weight > 0.3:
            self.hybrid_weight *= self.hybrid_decay
            self.hybrid_weight = max(0.3, self.hybrid_weight)

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        robot_grid = (int(robot.x // self.cell_size), int(robot.y // self.cell_size))

        # Loop detection
        repeat = self.position_history.get(robot_grid, 0)
        self.position_history[robot_grid] = repeat + 1
        loop_penalty = -5 * min(repeat, 5) if repeat > 2 else 0

        # Curiosity bonus
        visit = self.visit_counts.get(robot_grid, 0)
        self.visit_counts[robot_grid] = visit + 1
        curiosity = 0.5 / (1 + visit * 0.3)

        # Goal reached
        if reached_goal:
            time_bonus = max(0, 100 - self.steps_since_goal * 0.5)
            return 200 + time_bonus + curiosity

        # Collision
        if robot.check_collision(obstacles):
            return -150

        # Progress reward
        if prev_distance is not None:
            progress = prev_distance - distance_to_goal
            move_reward = progress * 20 if progress > 0 else progress * 10

            # Distance-based shaping
            distance_reward = -distance_to_goal * 0.01

            # Time penalty
            step_penalty = -0.1

            self.steps_since_goal += 1

            return move_reward + distance_reward + step_penalty + loop_penalty + curiosity

        return -0.1 + curiosity + loop_penalty

    def _save_model_implementation(self):
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'hybrid_weight': self.hybrid_weight,
            'step_count': self.step_count
        }
        torch.save(checkpoint, self.model_path)
        print(f"Model saved to {self.model_path}")

    def _load_model_implementation(self):
        checkpoint = torch.load(self.model_path)
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
        if 'step_count' in checkpoint:
            self.step_count = checkpoint['step_count']

        self.q_network.eval()
        print(f"Model loaded from {self.model_path}")