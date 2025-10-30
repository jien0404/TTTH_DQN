import pygame
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from Robot import Robot
from Obstacle import Obstacle
from controller.Controller import Controller
from controller.DQNController import DQNController
from controller.DFQL import DFQLController
from controller.DWAController import DWAController
from controller.DQN_PER_MTS import DQNPMController
from controller.SDQN import SDQNController
from controller.TransformerController import TransformerDQNController
from Colors import *
from MapData import maps

def get_mode():
    """Get mode from user input with validation"""
    while True:
        mode = input("Enter mode (train/test): ").lower()
        if mode in ["train", "test"]:
            return mode == "train"
        print("Invalid mode. Please enter 'train' or 'test'")

def select_controller():
    """Get controller selection from user input with validation"""
    print("\nAvailable controllers:")
    controllers = {
        1: "DQNController",
        2: "DFQLController",
        3: "DWAController",
        4: "DQNPMController",
        5: "SDQNController",
        6: "TransformerDQNController",
    }
    
    for idx, name in controllers.items():
        print(f"{idx}. {name}")
    
    while True:
        try:
            choice = int(input(f"Select controller (1-{len(controllers)}): "))
            if 1 <= choice <= len(controllers):
                controller_name = controllers[choice]
                print(f"Selected controller: {controller_name}")
                return controller_name
            print(f"Please enter a number between 1 and {len(controllers)}")
        except ValueError:
            print("Please enter a valid number")

# Khởi tạo Pygame
pygame.init()

# Thiết lập thông số môi trường
env_size = 512
cell_size = 16
env_padding = int(env_size * 0.06)  # ~30 pixel
GRID_WIDTH = env_size // cell_size  # 32 ô
GRID_HEIGHT = env_size // cell_size  # 32 ô
WINDOW_WIDTH = env_size + 2 * env_padding
WINDOW_HEIGHT = env_size + 2 * env_padding + 3 * env_padding  # Thêm không gian cho nút bấm và stats

# Thiết lập font
my_font = pygame.font.SysFont("arial", env_padding // 2)
stats_font = pygame.font.SysFont("arial", int(env_padding // 2.5))

# Chọn map
def select_map():
    print("Available maps:")
    for i, map_name in enumerate(maps.keys()):
        print(f"{i+1}. {map_name}")
    
    while True:
        try:
            choice = int(input(f"Select map (1-{len(maps)}): "))
            if 1 <= choice <= len(maps):
                return list(maps.keys())[choice-1]
            print(f"Please enter a number between 1 and {len(maps)}")
        except ValueError:
            print("Please enter a valid number")

class Environment:
    def __init__(self, name_map, is_training=True, selected_map="map1", controller_name="ClassicalQL", model_path="models/dqn_model.pth"):
        self.is_training = is_training
        self.model_path = model_path
        self.name_map = name_map
        self.controller_name = controller_name
        os.makedirs(f"models/{name_map}/{controller_name}", exist_ok=True)
        os.makedirs(f"plots/{name_map}/{controller_name}", exist_ok=True)
        os.makedirs(f"results/{name_map}/{controller_name}", exist_ok=True)
        
        # Initialize window
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Robot Navigation with DQN" + (" - Training" if is_training else " - Testing"))
        
        # Load map
        self.map_data = maps[selected_map]
        
        # Set up environment components
        self.setup_environment()
        
        # Training stats
        self.episode_count = 0
        self.step_count = 0
        self.reset_count = 0
        self.total_rewards = []
        self.episode_lengths = []
        self.best_reward = float('-inf')
        self.collision_count = 0
        self.goal_reached_count = 0
        self.trap_count = 0  # Đếm số lần robot rơi vào bẫy
        self.prev_distance_to_goal = None
        self.episode_reward = 0
        self.episode_steps = 0
        self.last_save_time = time.time()
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        self.FPS = 60 if not is_training else 240

    def setup_environment(self):
        # Get Start and Goal positions from map
        start_grid_x, start_grid_y = self.map_data["Start"]
        goal_grid_x, goal_grid_y = self.map_data["Goal"]
        
        # Convert grid positions to pixel coordinates
        self.start = (
            env_padding + (start_grid_x + 0.5) * cell_size,
            env_padding + (start_grid_y + 0.5) * cell_size
        )
        self.goal = (
            env_padding + (goal_grid_x + 0.5) * cell_size,
            env_padding + (goal_grid_y + 0.5) * cell_size
        )
        
        # Create obstacles from map data
        self.obstacles = []
        for obs_data in self.map_data["Obstacles"]:
            obstacle = Obstacle(
                x=obs_data["x"],
                y=obs_data["y"],
                width=obs_data["width"],
                height=obs_data["height"],
                static=obs_data["static"],
                velocity=obs_data.get("velocity"),
                x_bound=obs_data.get("x_bound"),
                y_bound=obs_data.get("y_bound"),
                path=obs_data.get("path"),
                angle=obs_data.get("angle", 0)
            )
            self.obstacles.append(obstacle)
        
        # Create controller and robot
        if self.controller_name == "DQNController":
            self.controller = DQNController(self.goal, cell_size, env_padding, is_training=self.is_training, model_path=self.model_path)
        elif self.controller_name == "DFQLController":
            self.controller = DFQLController(self.goal, cell_size, env_padding, is_training=self.is_training, model_path=self.model_path)
        elif self.controller_name == "DQNPMController":
            self.controller = DQNPMController(self.goal, cell_size, env_padding, is_training=self.is_training, model_path=self.model_path)
        elif self.controller_name == "SDQNController":
            self.controller = SDQNController(self.goal, cell_size, env_padding, is_training=self.is_training, model_path=self.model_path)
        elif self.controller_name == "DWAController":
            self.is_training = False  # DWAController only runs in test mode
            self.controller = DWAController(self.goal, cell_size, env_padding, is_training=self.is_training, model_path=self.model_path)
        elif self.controller_name == "TransformerDQNController":
            self.controller = TransformerDQNController(self.goal, cell_size, env_padding, is_training=self.is_training, model_path=self.model_path)
        else:
            # Default to ClassicalQL if controller not found
            print(f"Warning: Controller {self.controller_name} not recognized. Using ClassicalQL as default.")
            self.controller = DQNController(self.goal, cell_size, env_padding, is_training=self.is_training, model_path=self.model_path)
        
        # Load model if not DWAController and in test mode
        if self.controller_name != "DWAController" and not self.is_training:
            self.controller.load_model()
        
        self.robot = Robot(self.start[0], self.start[1], cell_size, self.controller, vision=cell_size*2.5, radius=8, env_padding=env_padding)
        
        # Robot path history
        self.robot_path = [(self.robot.x, self.robot.y)]
        # State history for trap detection
        self.state_history = []
        
        # UI elements
        self.button_start = pygame.Rect(env_padding + int(env_size * 0.7), env_padding * 2 + env_size,
                                int(env_size * 0.2), int(env_padding * 0.4))
        self.button_pause = pygame.Rect(env_padding + int(env_size * 0.4), env_padding * 2 + env_size,
                                int(env_size * 0.2), int(env_padding * 0.4))
        self.button_start_text = my_font.render("Start", True, BLACK)
        self.button_pause_text = my_font.render("Pause", True, BLACK)

    def reset_episode(self, reached_goal, trapped=False):
        """Reset the robot to the starting position and clear path history"""
        self.robot.x, self.robot.y = self.start
        self.robot.grid_x, self.robot.grid_y = self.map_data["Start"]
        self.robot_path.clear()
        self.robot_path.append((self.robot.x, self.robot.y))
        self.state_history.clear()  # Clear state history
        self.prev_distance_to_goal = None
        
        if self.is_training:
            self.total_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_steps)
            
            # Reset episode stats
            self.episode_count += 1
            self.episode_reward = 0
            self.episode_steps = 0
            
            # Update epsilon
            if self.controller_name in ["DQNController", "DQNPMController", "SDQNController", "TransformerDQNController"]:
                self.controller.update_epsilon()
            else:
                if reached_goal:
                    self.controller.update_epsilon()

    def draw_grid(self):
        """Draw the grid background"""
        self.window.fill(WHITE)
        for x in range(env_padding, WINDOW_WIDTH - env_padding, cell_size):
            pygame.draw.line(self.window, BLACK, (x, env_padding), (x, env_size + env_padding))
        for y in range(env_padding, env_size + env_padding, cell_size):
            pygame.draw.line(self.window, BLACK, (env_padding, y), (WINDOW_WIDTH - env_padding, y))

    def draw_stats(self):
        """Draw training/testing statistics"""
        stats_y = env_padding * 3 + env_size
        stats_x = env_padding
        line_height = env_padding // 3
        
        texts = []
        if self.is_training:
            texts = [
                f"Episodes: {self.episode_count}",
                f"Reset Count: {self.reset_count}",
                f"Epsilon: {self.controller.epsilon:.4f}",
                f"Goals: {self.goal_reached_count}",
                f"Collisions: {self.collision_count}"
            ]
            
            if len(self.total_rewards) > 0:
                avg_reward = sum(self.total_rewards[-100:]) / min(len(self.total_rewards), 100)
                texts.append(f"Avg Reward (100): {avg_reward:.2f}")
        else:
            texts = [
                f"Testing Mode",
                f"Goals: {self.goal_reached_count}",
                f"Collisions: {self.collision_count}",
                f"Trapped: {self.trap_count}"  # Thêm thống kê số lần bị kẹt
            ]
        
        for i, text in enumerate(texts):
            text_surface = stats_font.render(text, True, BLACK)
            self.window.blit(text_surface, (stats_x, stats_y + i * line_height))

    def draw_environment(self):
        """Draw the complete environment"""
        # Draw grid
        self.draw_grid()
        
        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(self.window)
        
        # Draw robot path
        if len(self.robot_path) > 1:
            pygame.draw.lines(self.window, RED, False, self.robot_path, 2)
        
        # Draw start and goal
        pygame.draw.rect(self.window, GREEN, (self.start[0]-5, self.start[1]-5, 10, 10))
        pygame.draw.circle(self.window, RED, self.goal, self.robot.radius)
        
        # Draw robot
        self.robot.draw(self.window)
        
        # Draw border
        pygame.draw.rect(self.window, BLACK, (env_padding, env_padding, env_size, env_size), 3)
        
        # Draw buttons
        pygame.draw.rect(self.window, BLACK, self.button_start, 4)
        pygame.draw.rect(self.window, BLACK, self.button_pause, 4)
        self.window.blit(self.button_start_text, self.button_start_text.get_rect(center=self.button_start.center))
        self.window.blit(self.button_pause_text, self.button_pause_text.get_rect(center=self.button_pause.center))
        
        # Draw stats
        self.draw_stats()

    def update(self):
        """Update environment state for one step"""
        # Move obstacles
        for obstacle in self.obstacles:
            obstacle.move()

        # Get current state
        state = self.robot.get_state(self.obstacles, GRID_WIDTH, GRID_HEIGHT, self.goal)
        _, current_distance_to_goal = state

        # Move robot
        direction = self.controller.make_decision(self.robot, self.obstacles)
        # Lưu lại hướng di chuyển để sử dụng trong training
        action_idx = self.controller.directions.index(direction)

        # Di chuyển robot và kiểm tra va chạm
        collided = self.robot.move(self.obstacles, GRID_WIDTH, GRID_HEIGHT)

        # Update path history
        if (self.robot.x, self.robot.y) != self.robot_path[-1]:
            self.robot_path.append((self.robot.x, self.robot.y))
            if len(self.robot_path) > 1000:  # Limit path length
                self.robot_path.pop(0)

        # Get next state
        next_state = self.robot.get_state(self.obstacles, GRID_WIDTH, GRID_HEIGHT, self.goal)
        _, next_distance_to_goal = next_state

        # Use grid-based distance returned by get_state (units: cells)
        # Consider reached if <= 1 cell (inclusive)
        reached_goal = next_distance_to_goal <= 1.0

        # Determine done
        done = reached_goal or collided

        if reached_goal:
            # snap robot to exact goal position to avoid small oscillations
            self.robot.x, self.robot.y = self.goal
            self.goal_reached_count += 1
        if collided:
            self.collision_count += 1

        # Update training information if in training mode
        if self.is_training:
            reward = self.controller.calculate_reward(
                self.robot, 
                self.obstacles, 
                done, 
                reached_goal, 
                next_distance_to_goal, 
                prev_distance=current_distance_to_goal
            )

            self.controller.store_experience(state, action_idx, reward, next_state, done)
            self.controller.train()

            # Update episode stats
            self.episode_reward += reward
            self.episode_steps += 1
            self.step_count += 1

        # Update previous distance
        self.prev_distance_to_goal = next_distance_to_goal

        print(f"Robot pos: {(self.robot.x, self.robot.y)}, Goal: {self.goal}, dist: {next_distance_to_goal:.2f}, reached_goal: {reached_goal}")

        return done, reached_goal, collided

    def save_model(self):
        """Save model and training statistics"""
        # Save model
        self.controller.save_model()
        
        # Save metrics if we have enough data
        if len(self.total_rewards) > 1:
            self.plot_metrics()
        
        print(f"Model saved at episode {self.episode_count}")

    def plot_metrics(self, filename=None):
        """Plot and save training metrics.
        Nếu filename không cung cấp thì ưu tiên self.run_plot_file (tạo ở train()).
        """
        plt.figure(figsize=(12, 8))

        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(self.total_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        # Plot episode lengths
        plt.subplot(2, 1, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        plt.tight_layout()

        if filename is None:
            filename = getattr(self, "run_plot_file", None)
            if filename is None:
                plot_dir = f"plots/{self.name_map}/{self.controller_name}"
                os.makedirs(plot_dir, exist_ok=True)
                filename = f"{plot_dir}/training_metrics_latest.png"

        plt.savefig(filename)
        plt.close()
        print(f"Saved training plot: {filename}")

    def train(self, max_episodes=1000):
        """Run training loop for specified number of episodes"""
        # --- Tạo folder duy nhất cho lần chạy training này ---
        plot_root = f"plots/{self.name_map}/{self.controller_name}"
        os.makedirs(plot_root, exist_ok=True)
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_plot_dir = os.path.join(plot_root, run_ts)
        os.makedirs(self.run_plot_dir, exist_ok=True)
        # file sẽ luôn có tên này trong mỗi run (ghi đè nếu plot_metrics được gọi nhiều lần)
        self.run_plot_file = os.path.join(self.run_plot_dir, "training_metrics_latest.png")

        running = True
        started = True
        pause = False
        
        print(f"Training for {max_episodes} episodes...")
        
        while running and self.episode_count < max_episodes:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if self.button_start.collidepoint(mouse_x, mouse_y):
                        started = True
                    elif self.button_pause.collidepoint(mouse_x, mouse_y):
                        pause = not pause
                        
            # Update environment if started and not paused
            if started and not pause:
                done, reached_goal, collided = self.update()

                # Nếu episode kết thúc (về đích hoặc va chạm), reset và lưu thông tin cần thiết
                if done:
                    self.reset_count += 1

                    # Gọi reset để cập nhật stats + clear path/history
                    # (trapped không áp dụng ở training, truyền False)
                    self.reset_episode(reached_goal=reached_goal, trapped=False)

                    # Save model định kỳ (giữ nguyên cơ chế cũ)
                    current_time = time.time()
                    if current_time - self.last_save_time > 60:  # Save every minute
                        self.save_model()
                        self.last_save_time = current_time

            # Draw environment
            self.draw_environment()
            pygame.display.update()

            # Control FPS
            self.clock.tick(self.FPS)
        
        # Final save
        self.save_model()
        print("Training completed!")
        
        # Plot final metrics
        self.plot_metrics()
        
        return self.controller.model_path

    def save_path_to_file(self, episode_number):
        """Lưu đường đi hiện tại của robot vào file text"""
        # Tạo thư mục results nếu chưa tồn tại
        os.makedirs(f"results/{self.name_map}/{self.controller_name}", exist_ok=True)

        # Tìm tên bản đồ bằng cách so sánh dữ liệu
        map_name = "unknown"
        for name, data in maps.items():
            if data == self.map_data:
                map_name = name
                break
            
        # Tạo tên file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/{self.name_map}/{self.controller_name}/robot_path_{map_name}_{timestamp}_ep{episode_number}.txt"

        # Lưu tọa độ đường đi
        with open(filename, "w") as f:
            for x, y in self.robot_path:
                # Chuyển đổi tọa độ pixel sang tọa độ lưới để dễ phân tích
                grid_x = (x - env_padding) // cell_size
                grid_y = (y - env_padding) // cell_size
                f.write(f"{grid_x},{grid_y}\n")

        print(f"Đã lưu đường đi vào {filename}")

    def test(self, name_map, episodes=10):
        """Chạy vòng lặp kiểm thử với số lượng episodes cụ thể"""
        running = True
        episode_count = 0
        steps_per_episode = []
        current_steps = 0

        # Tạo thư mục results/map/controller nếu chưa tồn tại
        result_dir = f"results/{name_map}/{self.controller_name}"
        os.makedirs(result_dir, exist_ok=True)

        # Tạo tên file chung cho tất cả các episode
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = f"{result_dir}/robot_paths_{timestamp}.txt"

        # Tạo danh sách để theo dõi đường đi và lịch sử trạng thái
        current_episode_path = []
        self.state_history = []

        print(f"Kiểm thử cho {episodes} episodes...")

        # Khởi tạo đường đi ban đầu
        current_episode_path.append((self.robot.x, self.robot.y))
        init_gx = int((self.robot.x - env_padding) // cell_size)
        init_gy = int((self.robot.y - env_padding) // cell_size)
        self.state_history.append((init_gx, init_gy))

        while running and episode_count < episodes:
            # Xử lý sự kiện
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Cập nhật vị trí robot trước khi gọi update()
            old_x, old_y = self.robot.x, self.robot.y

            # Cập nhật môi trường
            done, reached_goal, collided = self.update()
            current_steps += 1

            # Lưu tọa độ lưới thay vì pixel để tránh so sánh float
            grid_x = int((self.robot.x - env_padding) // cell_size)
            grid_y = int((self.robot.y - env_padding) // cell_size)
            current_grid_state = (grid_x, grid_y)

            # Kiểm tra nếu robot đã di chuyển, thêm vị trí mới vào đường đi và lịch sử trạng thái
            if (self.robot.x, self.robot.y) != (old_x, old_y):
                current_episode_path.append((self.robot.x, self.robot.y))
                # append grid state (ints) — để đếm lặp ổn định
                self.state_history.append(current_grid_state)
                if len(self.state_history) > 100:
                    self.state_history.pop(0)
            else:
                print(f"Bị kẹt tại: {self.robot.x}, {self.robot.y}.")

            # Kiểm tra chu kỳ trạng thái (bẫy) — CHỈ khi chưa reach goal và chưa va chạm
            trapped = False
            if not reached_goal and not collided and len(self.state_history) >= 10:
                state_counts = self.state_history.count(current_grid_state)
                if state_counts >= 3:  # Lặp lại 3 lần được coi là bẫy
                    trapped = True
                    self.trap_count += 1
                    done = True

            if done:
                # Lưu đường đi của episode hiện tại vào file
                with open(result_file, "a") as f:
                    outcome = "Trapped" if trapped else ("Goal" if reached_goal else "Collision")
                    f.write(f"# Episode {episode_count + 1} ({outcome})\n")
                    for x, y in current_episode_path:
                        gx = int((x - env_padding) // cell_size)
                        gy = int((y - env_padding) // cell_size)
                        f.write(f"[{gx},{gy}], ")
                    f.write("\n")  # Dòng trống giữa các episode

                # Reset đường đi và lịch sử trạng thái cho episode tiếp theo
                current_episode_path = [(self.start[0], self.start[1])]
                # Gọi reset 1 lần tại đây — truyền reached_goal chính xác và trapped flag
                self.reset_episode(reached_goal=reached_goal, trapped=trapped)

                episode_count += 1
                steps_per_episode.append(current_steps)
                current_steps = 0
                print(f"Episode {episode_count}/{episodes} hoàn thành và đã lưu đường đi")

            # Vẽ môi trường
            self.draw_environment()
            pygame.display.update()

            # Kiểm soát FPS 
            self.clock.tick(self.FPS)

        # In kết quả kiểm thử
        success_rate = self.goal_reached_count / max(1, episode_count) * 100
        avg_steps = sum(steps_per_episode) / max(1, len(steps_per_episode))

        print("\nKết quả kiểm thử:")
        print(f"Episodes: {episode_count}")
        print(f"Đạt mục tiêu: {self.goal_reached_count} ({success_rate:.2f}%)")
        print(f"Va chạm: {self.collision_count}")
        print(f"Bị kẹt: {self.trap_count}")
        print(f"Số bước trung bình: {avg_steps:.2f}")
        print(f"Tất cả đường đi đã được lưu vào file: {result_file}")

        return success_rate, avg_steps

def main():
    # Get mode from user
    is_training = get_mode()

    controller_name = select_controller()

    selected_map = select_map()
    
    if is_training:
        # Configure training
        model_name = f"dqn_{selected_map}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        # Thay đổi đường dẫn model để phân theo controller
        model_path = os.path.join(f"models/{selected_map}/{controller_name}", model_name)
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        max_episodes = int(input("Enter number of training episodes (default 1000): ") or "1000")
        
        # Initialize and run training
        env = Environment(is_training=True, name_map=selected_map, selected_map=selected_map, controller_name=controller_name, model_path=model_path)
        model_path = env.train(max_episodes=max_episodes)
        
        # Ask if user wants to test the trained model
        test_after = input("Test the trained model? (y/n): ").lower() == 'y'
        if test_after:
            env = Environment(is_training=False, name_map=selected_map, selected_map=selected_map, controller_name=controller_name, model_path=model_path)
            env.test(selected_map, episodes=5)
    else:
        # Skip model selection for DWAController
        if controller_name == "DWAController":
            model_path = None
            env = Environment(is_training=False, name_map=selected_map, selected_map=selected_map, controller_name=controller_name, model_path=model_path)
            test_episodes = int(input("Enter number of test episodes (default 10): ") or "10")
            env.test(selected_map, episodes=test_episodes)
        else:
            # List available models
            models_dir = f"models/{selected_map}/{controller_name}"
            if not os.path.exists(models_dir) or not os.listdir(models_dir):
                print("No trained models found. Please train a model first.")
                return
            
            print("\nAvailable models:")
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")
            
            # Select model
            while True:
                try:
                    choice = int(input(f"Select model (1-{len(model_files)}): "))
                    if 1 <= choice <= len(model_files):
                        model_path = os.path.join(models_dir, model_files[choice-1])
                        break
                    print(f"Please enter a number between 1 and {len(model_files)}")
                except ValueError:
                    print("Please enter a valid number")
            
            # Configure and run testing
            test_episodes = int(input("Enter number of test episodes (default 10): ") or "10")
            env = Environment(is_training=False, name_map=selected_map, selected_map=selected_map, controller_name=controller_name, model_path=model_path)
            env.test(selected_map, episodes=test_episodes)

    pygame.quit()

if __name__ == "__main__":
    main()