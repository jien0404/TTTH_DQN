# main_classical.py

import pygame
import numpy as np
import os
import time
from Robot import Robot
from Obstacle import Obstacle
from controller.WaitingController import WaitingController # Import controller mới
from controller.IndoorAdaptedController import IndoorAdaptedController

from Colors import *
from MapData import maps

# ==============================================================================
# CÁC THAM SỐ VÀ HÀM TIỆN ÍCH (Giống main.py cũ)
# ==============================================================================
pygame.init()
env_size = 512
cell_size = 16
env_padding = int(env_size * 0.06)
GRID_WIDTH = env_size // cell_size
GRID_HEIGHT = env_size // cell_size
WINDOW_WIDTH = env_size + 2 * env_padding
WINDOW_HEIGHT = env_size + 2 * env_padding + 3 * env_padding

def select_map():
    print("Available maps:")
    map_keys = list(maps.keys())
    for i, map_name in enumerate(map_keys):
        print(f"{i+1}. {map_name}")
    while True:
        try:
            choice = int(input(f"Select map (1-{len(map_keys)}): "))
            if 1 <= choice <= len(map_keys): return map_keys[choice-1]
        except ValueError: print("Please enter a valid number")

# ==============================================================================
# MÔI TRƯỜNG MÔ PHỎNG (Phiên bản rút gọn cho test)
# ==============================================================================

class TestEnvironment:
    def __init__(self, selected_map, controller_name):
        self.map_name = selected_map
        self.controller_name = controller_name
        
        # Tạo thư mục lưu kết quả nếu chưa có
        os.makedirs(f"results/{self.map_name}/{self.controller_name}", exist_ok=True)
        
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f"Testing: {controller_name} on {selected_map}")
        
        self.map_data = maps[selected_map]
        self.setup_environment()
        
        # Stats
        self.collision_count = 0
        self.goal_reached_count = 0
        self.trap_count = 0
        
        self.clock = pygame.time.Clock()
        self.FPS = 10

    def setup_environment(self):
        start_gx, start_gy = self.map_data["Start"]
        goal_gx, goal_gy = self.map_data["Goal"]
        
        self.start = (env_padding + (start_gx + 0.5) * cell_size, env_padding + (start_gy + 0.5) * cell_size)
        self.goal = (env_padding + (goal_gx + 0.5) * cell_size, env_padding + (goal_gy + 0.5) * cell_size)
        
        self.obstacles = [Obstacle(**obs_data) for obs_data in self.map_data["Obstacles"]]
        
        # Khởi tạo controller được chỉ định
        if self.controller_name == "WaitingController":
            self.controller = WaitingController(self.goal, cell_size, env_padding, GRID_WIDTH, GRID_HEIGHT)
        elif self.controller_name == "IndoorAdaptedController":
            self.controller = IndoorAdaptedController(
                self.goal, cell_size, env_padding, GRID_WIDTH, GRID_HEIGHT
            )
        else:
            raise ValueError(f"Controller {self.controller_name} not supported in this script.")
            
        self.robot = Robot(self.start[0], self.start[1], cell_size, self.controller, vision=cell_size * 2.5, radius=6, env_padding=env_padding)

    def reset_episode(self):
        self.robot.x, self.robot.y = self.start
        self.robot.grid_x, self.robot.grid_y = self.map_data["Start"]
        # Reset trạng thái nội bộ của controller nếu có
        if hasattr(self.controller, 'reset'):
            self.controller.reset()

    def draw_environment(self, robot_path):
        self.window.fill(WHITE)
        # Vẽ lưới
        for x in range(env_padding, env_padding + env_size + 1, cell_size):
            pygame.draw.line(self.window, BLACK, (x, env_padding), (x, env_size + env_padding))
        for y in range(env_padding, env_padding + env_size + 1, cell_size):
            pygame.draw.line(self.window, BLACK, (env_padding, y), (env_padding + env_size, y))
        # Vẽ các thành phần khác
        for obs in self.obstacles: obs.draw(self.window)
        if len(robot_path) > 1: pygame.draw.lines(self.window, RED, False, robot_path, 2)
        pygame.draw.rect(self.window, GREEN, (self.start[0]-5, self.start[1]-5, 10, 10))
        pygame.draw.circle(self.window, RED, self.goal, self.robot.radius)
        self.robot.draw(self.window)
        pygame.draw.rect(self.window, BLACK, (env_padding, env_padding, env_size, env_size), 3)

    def run_test(self, num_episodes=10):
        print(f"Running test for {num_episodes} episodes...")
        
        # Tạo file kết quả duy nhất cho lần chạy này
        result_dir = f"results/{self.map_name}/{self.controller_name}"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file_path = os.path.join(result_dir, f"robot_paths_{timestamp}.txt")

        all_steps = []
        
        for ep in range(num_episodes):
            self.reset_episode()
            robot_path = [(self.robot.x, self.robot.y)]
            state_history = [(self.robot.grid_x, self.robot.grid_y)]
            current_steps = 0
            
            running_episode = True
            while running_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Di chuyển vật cản động
                for obs in self.obstacles: obs.move()
                
                # Ra quyết định và di chuyển
                collided = self.robot.move(self.obstacles, GRID_WIDTH, GRID_HEIGHT)
                
                # Ghi lại đường đi
                robot_path.append((self.robot.x, self.robot.y))
                state_history.append((self.robot.grid_x, self.robot.grid_y))
                current_steps += 1

                # Kiểm tra điều kiện kết thúc
                reached_goal = np.linalg.norm(np.array((self.robot.x, self.robot.y)) - np.array(self.goal)) < self.robot.radius + 5
                
                trapped = False
                if len(state_history) > 100 and len(set(state_history[-20:])) <= 3:
                    trapped = True
                    self.trap_count += 1
                
                done = reached_goal or collided or trapped or current_steps > 2000 # Thêm timeout

                if done:
                    if reached_goal: self.goal_reached_count += 1
                    if collided: self.collision_count += 1
                    all_steps.append(current_steps)
                    
                    # Ghi kết quả vào file (ĐỒNG BỘ ĐỊNH DẠNG)
                    with open(result_file_path, "a") as f:
                        outcome = "Goal" if reached_goal else ("Trapped" if trapped else "Collision")
                        f.write(f"# Episode {ep + 1} ({outcome})\n")
                        path_str = ", ".join([f"[{int((p[0]-env_padding)//cell_size)},{int((p[1]-env_padding)//cell_size)}]" for p in robot_path])
                        f.write(path_str + "\n")
                    
                    print(f"Episode {ep + 1}/{num_episodes} finished. Outcome: {outcome}, Steps: {current_steps}")
                    running_episode = False

                # Vẽ
                self.draw_environment(robot_path)
                pygame.display.flip()
                self.clock.tick(self.FPS)

        # In kết quả cuối cùng
        print("\n--- Test Summary ---")
        print(f"Total Episodes: {num_episodes}")
        print(f"Success Rate: {self.goal_reached_count / num_episodes * 100:.2f}%")
        print(f"Collisions: {self.collision_count}")
        print(f"Trapped: {self.trap_count}")
        print(f"Average Steps: {np.mean(all_steps):.2f}")
        print(f"Results saved to: {result_file_path}")
        pygame.quit()

def main():
    selected_map = select_map()
    controller_name = "IndoorAdaptedController" # Chỉ chạy controller này
    
    num_episodes = int(input("Enter number of test episodes (default 10): ") or "10")

    env = TestEnvironment(selected_map, controller_name)
    env.run_test(num_episodes)

if __name__ == "__main__":
    main()