import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

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
        # Phần chung
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Nhánh giá trị trạng thái V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(64, 1)
        )
        # Nhánh lợi thế hành động A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Kết hợp V(s) và A(s, a) để tính Q(s, a)
        if len(advantage.shape) == 1:  # Single sample case (no batch dimension)
            q = value + advantage - advantage.mean(dim=0, keepdim=True)
        else:  # Batch case
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q



class Controller:
    def __init__(self, goal, cell_size, env_padding, is_training=True, model_path="dqn_model.pth"):
        self.goal = goal
        self.cell_size = cell_size
        self.env_padding = env_padding
        self.is_training = is_training
        self.model_path = model_path

        # Định nghĩa 8 hướng rời rạc
        self.directions = [
            (1, 0),   # Đông
            (1, 1),   # Đông Bắc
            (0, 1),   # Bắc
            (-1, 1),  # Tây Bắc
            (-1, 0),  # Tây
            (-1, -1), # Tây Nam
            (0, -1),  # Nam
            (1, -1)   # Đông Nam
        ]

        # Thiết lập DQN
        self.state_dim = 5 * 5 + 1  # Ma trận 5x5 (25) + khoảng cách đến đích (1) = 26 chiều
        self.action_dim = len(self.directions)  # 8 hướng
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sử dụng CPU vì môi trường không cần GPU
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        self.gamma = 0.99  # Hệ số chiết khấu phần thưởng
        self.epsilon = 1.0 if is_training else 0.0  # Epsilon-greedy: 1.0 khi train, 0.0 khi test
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.batch_size = 64
        self.memory = deque(maxlen=20000)  # Bộ nhớ Experience Replay
        self.memory.clear()  # Xóa dữ liệu cũ
        self.target_update_freq = 200  # Cập nhật target network sau mỗi 100 bước
        self.step_count = 0
        self.metrics = {
                'episode_rewards': [],
                'episode_lengths': [],
                'avg_q_values': []
            }
        # Tránh bẫy
        self.position_history = {}  # Dict để đếm số lần robot đã ở mỗi vị trí
        self.position_memory_size = 100  # Chỉ nhớ 100 vị trí gần nhất
        self.position_history_list = []  # Danh sách các vị trí theo thứ tự truy cập

        self.visit_counts = {}
        self.curiosity_factor = 0.1

        print("gọi cha")

        # Tải mô hình nếu ở chế độ test
        # if not self.is_training:
        #     self.load_model()

    def make_decision(self, robot, obstacles):
        # Lấy trạng thái hiện tại (ma trận 5x5 và khoảng cách đến đích)
        state, distance_to_goal = robot.get_state(obstacles, 32, 32, self.goal)  # Giả sử lưới 32x32
        # Kết hợp ma trận trạng thái và khoảng cách đến đích thành một vector
        state_flat = state.flatten()
        combined_state = np.concatenate([state_flat, [distance_to_goal]])
        state_tensor = torch.FloatTensor(combined_state).to(self.device)

        # Cấu hình chế độ cho mạng Q
        if self.is_training:
            self.q_network.train()
        else:
            self.q_network.eval()

        # Epsilon-greedy: Chọn hành động ngẫu nhiên hoặc dựa trên Q-value
        if random.random() < self.epsilon and self.is_training:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()

        # Trả về hướng tương ứng
        return self.directions[action_idx]

    def store_experience(self, state, action_idx, reward, next_state, done):
        # state và next_state giờ là tuple (ma trận 5x5, khoảng cách đến đích)
        state_matrix, state_distance = state
        next_state_matrix, next_state_distance = next_state
        # Kết hợp ma trận trạng thái và khoảng cách đến đích
        state_flat = state_matrix.flatten()
        next_state_flat = next_state_matrix.flatten()
        state_combined = np.concatenate([state_flat, [state_distance]])
        next_state_combined = np.concatenate([next_state_flat, [next_state_distance]])
        # Chuyển thành tensor
        state_tensor = torch.FloatTensor(state_combined).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state_combined).to(self.device)
        action = torch.LongTensor([action_idx]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        self.memory.append((state_tensor, action, reward, next_state_tensor, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Lấy mẫu ngẫu nhiên từ bộ nhớ
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards).squeeze()  # Ép về [batch_size]
        next_states = torch.stack(next_states)
        dones = torch.stack(dones).squeeze()  # Ép về [batch_size]

        # Tính Q-value hiện tại
        q_values = self.q_network(states).gather(1, actions)

        # Tính Q-value mục tiêu sử dụng DDQN
        with torch.no_grad():
            # Sử dụng mạng Q chính để chọn hành động tốt nhất tại trạng thái tiếp theo
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)

            # Sử dụng mạng mục tiêu để đánh giá giá trị của hành động đã chọn
            next_q_values = self.target_network(next_states).gather(1, next_actions)

            # Tính toán Q-value mục tiêu
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.squeeze()


        # Tính loss và cập nhật mạng
        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Cập nhật target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, robot, obstacles, done, reached_goal, distance_to_goal, prev_distance=None):
        # Lưu vị trí hiện tại và kiểm tra trùng lặp
        position = (robot.grid_x, robot.grid_y)
        
        # Cập nhật lịch sử vị trí
        if position in self.position_history:
            self.position_history[position] += 1
        else:
            self.position_history[position] = 1
        
        # Thêm vào danh sách vị trí
        self.position_history_list.append(position)
        
        # Giữ kích thước danh sách trong giới hạn
        if len(self.position_history_list) > self.position_memory_size:
            old_pos = self.position_history_list.pop(0)
            self.position_history[old_pos] -= 1
            if self.position_history[old_pos] <= 0:
                del self.position_history[old_pos]
        
        # Phạt cho việc lặp lại vị trí
        repetition_penalty = min(-2 * (self.position_history[position] - 1), 0)
        
        # Cập nhật visit counts cho curiosity
        if position in self.visit_counts:
            self.visit_counts[position] += 1
        else:
            self.visit_counts[position] = 1
        
        # Phần thưởng curiosity
        curiosity_reward = self.curiosity_factor / max(1, self.visit_counts[position]**0.5)

        # Phần thưởng cơ bản
        if reached_goal:
            return 100 + curiosity_reward  # Phần thưởng lớn khi đến đích
            
        if robot.check_collision(obstacles):
            return -50  # Phạt khi va chạm
        
         # Phần thưởng khi tiến gần hơn đến mục tiêu
        if prev_distance is not None:
            # Phần thưởng lớn hơn cho việc tiến gần đích
            progress_reward = (prev_distance - distance_to_goal) * 10
            
            # Phạt nhẹ cho mỗi bước đi để khuyến khích tìm đường ngắn nhất
            step_penalty = -0.1
            
            # Phạt mạnh cho việc đi lùi (xa đích hơn)
            if distance_to_goal > prev_distance:
                progress_reward -= 5  # Phạt thêm cho việc đi lùi
                
            return progress_reward + step_penalty + repetition_penalty + curiosity_reward
        
        # Nếu không có prev_distance
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