import pygame
import os
import numpy as np
from datetime import datetime
from MapData import maps
from Obstacle import Obstacle
from Colors import *

# ==============================================================================
# CÁC THAM SỐ VÀ HÀM TIỆN ÍCH
# ==============================================================================

# Sao chép các tham số môi trường từ main.py
env_size = 512
cell_size = 16
env_padding = int(env_size * 0.06)

# MỚI: Thêm không gian cho phần chú thích ở dưới
LEGEND_AREA_HEIGHT = 120
WINDOW_WIDTH = env_size + 2 * env_padding
WINDOW_HEIGHT = env_size + 2 * env_padding + LEGEND_AREA_HEIGHT

def select_map():
    """Cho phép người dùng chọn một bản đồ để phân tích."""
    print("Available maps:")
    map_keys = list(maps.keys())
    for i, map_name in enumerate(map_keys):
        print(f"{i+1}. {map_name}")
    
    while True:
        try:
            choice = int(input(f"Select map to visualize (1-{len(map_keys)}): "))
            if 1 <= choice <= len(map_keys):
                return map_keys[choice-1]
            print(f"Please enter a number between 1 and {len(map_keys)}")
        except ValueError:
            print("Please enter a valid number")

def calculate_path_length(path):
    """Tính độ dài đường đi thực tế dựa trên tọa độ pixel."""
    length = 0
    for i in range(1, len(path)):
        p1 = np.array(path[i-1])
        p2 = np.array(path[i])
        length += np.linalg.norm(p2 - p1)
    return length

def read_all_paths_and_outcomes(file_path):
    """Đọc TẤT CẢ các đường đi và kết quả (outcome) từ file."""
    all_runs = [] # Sẽ chứa các tuple (path, outcome)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("# Episode"):
                outcome = "Unknown"
                if "(Goal)" in line or "(Success)" in line:
                    outcome = "Success"
                elif "(Trapped)" in line:
                    outcome = "Trapped"
                elif "(Collision)" in line:
                    outcome = "Collision"

                if i + 1 < len(lines):
                    path_line = lines[i+1].strip()
                    try:
                        if path_line.endswith(','):
                            path_line = path_line[:-1]
                        grid_coords = eval(f"[{path_line}]")
                        pixel_coords = [(env_padding + (gx + 0.5) * cell_size, env_padding + (gy + 0.5) * cell_size) for gx, gy in grid_coords]
                        all_runs.append({'path': pixel_coords, 'outcome': outcome})
                    except Exception as e:
                        print(f"Warning: Could not parse path line in {os.path.basename(file_path)}. Line: '{path_line}', Error: {e}")
    return all_runs

def find_best_effort_path(controller_path):
    """
    Tìm đường đi tốt nhất:
    - Nếu có đường đi thành công -> Lấy đường ngắn nhất.
    - Nếu không có -> Lấy đường đi thất bại dài nhất.
    """
    successful_paths = []
    failed_paths = []

    path_files = [f for f in os.listdir(controller_path) if f.startswith("robot_paths_") and f.endswith(".txt")]
    for file_name in path_files:
        file_path = os.path.join(controller_path, file_name)
        runs = read_all_paths_and_outcomes(file_path)
        for run in runs:
            if run['outcome'] == 'Success':
                successful_paths.append(run['path'])
            else:
                failed_paths.append(run['path'])

    if successful_paths:
        # Tìm đường đi thành công ngắn nhất
        best_path = min(successful_paths, key=calculate_path_length)
        return best_path, "Success"
    elif failed_paths:
        # Nếu không thành công, tìm đường đi thất bại dài nhất
        best_effort_path = max(failed_paths, key=calculate_path_length)
        return best_effort_path, "Failed"
    
    return None, None

# ==============================================================================
# CLASS VISUALIZER CHÍNH
# ==============================================================================

class PathVisualizer:
    def __init__(self, map_name, paths_data):
        self.map_name = map_name
        self.paths_data = paths_data
        self.map_data = maps[map_name]
        
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f"Best Path Comparison on Map: {map_name}")
        self.font = pygame.font.SysFont("arial", 18)
        
        self.obstacles = [Obstacle(**obs_data) for obs_data in self.map_data["Obstacles"]]
        self.colors = [RED, GREEN, YELLOW]

    def draw_environment_base(self):
        # Nền trắng cho toàn bộ cửa sổ
        self.window.fill(WHITE)
        
        # Vẽ lưới chỉ trong khu vực bản đồ
        for x in range(env_padding, env_padding + env_size + 1, cell_size):
            pygame.draw.line(self.window, BLACK, (x, env_padding), (x, env_size + env_padding))
        for y in range(env_padding, env_padding + env_size + 1, cell_size):
            pygame.draw.line(self.window, BLACK, (env_padding, y), (env_padding + env_size, y))
        
        for obs in self.obstacles:
            obs.draw(self.window)
            
        start_gx, start_gy = self.map_data["Start"]
        goal_gx, goal_gy = self.map_data["Goal"]
        
        start_px = (env_padding + (start_gx + 0.5) * cell_size, env_padding + (start_gy + 0.5) * cell_size)
        goal_px = (env_padding + (goal_gx + 0.5) * cell_size, env_padding + (goal_gy + 0.5) * cell_size)

        pygame.draw.rect(self.window, GREEN, (start_px[0]-5, start_px[1]-5, 10, 10))
        pygame.draw.circle(self.window, RED, goal_px, 8)
        pygame.draw.rect(self.window, BLACK, (env_padding, env_padding, env_size, env_size), 3)

    def draw_paths_and_legend(self):
        # Khu vực chú thích bắt đầu bên dưới bản đồ
        legend_y_start = env_padding + env_size + 20
        legend_x_start = env_padding
        color_index = 0
        
        for controller_name, data in self.paths_data.items():
            path = data['path']
            status = data['status']
            length = data['length']
            
            if not path: continue
            
            color = self.colors[color_index % len(self.colors)]
            
            # Vẽ đường đi
            if len(path) > 1:
                pygame.draw.lines(self.window, color, False, path, 3)
            
            # Vẽ chú thích (legend)
            status_text = "(Best Attempt)" if status == "Failed" else ""
            legend_text = f"{controller_name} - Length: {length:.1f} {status_text}"
            text_surface = self.font.render(legend_text, True, BLACK)
            
            pygame.draw.line(self.window, color, (legend_x_start, legend_y_start + 9), (legend_x_start + 20, legend_y_start + 9), 4)
            self.window.blit(text_surface, (legend_x_start + 25, legend_y_start))
            
            legend_y_start += 25
            color_index += 1
            
    def run_and_save(self):
        self.draw_environment_base()
        self.draw_paths_and_legend()
        
        output_dir = "result_visualized"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"best_paths_{self.map_name}_{timestamp}.png")
        pygame.image.save(self.window, output_path)
        print(f"\nSaved visualization to: {output_path}")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.flip()
        
        pygame.quit()

# ==============================================================================
# HÀM MAIN ĐỂ CHẠY SCRIPT
# ==============================================================================

def main():
    selected_map = select_map()
    base_results_path = f"results/{selected_map}"

    if not os.path.exists(base_results_path):
        print(f"Error: Results directory not found for map '{selected_map}'")
        return

    controller_folders = [d for d in os.listdir(base_results_path) if os.path.isdir(os.path.join(base_results_path, d))]
    
    paths_to_visualize = {}

    print("\nSearching for best effort paths for each controller...")
    print("-" * 50)

    for controller_name in controller_folders:
        controller_path = os.path.join(base_results_path, controller_name)
        path, status = find_best_effort_path(controller_path)
        
        if path:
            paths_to_visualize[controller_name] = {
                'path': path,
                'status': status,
                'length': calculate_path_length(path)
            }
            print(f"Found path for: {controller_name} (Status: {status})")
        else:
            print(f"No paths found for: {controller_name}")

    if not paths_to_visualize:
        print("\nNo paths found for any controller. Cannot generate visualization.")
        return

    visualizer = PathVisualizer(selected_map, paths_to_visualize)
    visualizer.run_and_save()

if __name__ == "__main__":
    main()