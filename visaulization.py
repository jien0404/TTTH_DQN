import pygame
import os
import numpy as np
from pygame.locals import *
from MapData import maps
from Colors import *
from datetime import datetime
from Obstacle import Obstacle

def draw_target(window, target):
    """Vẽ điểm mục tiêu (đỏ)."""
    pygame.draw.circle(window, RED, target, 8, 0)

def draw_start(window, start):
    """Vẽ điểm bắt đầu (xanh lá)."""
    pygame.draw.circle(window, GREEN, start, 6, 0)

def draw_path(window, path, color):
    """Vẽ đường đi với màu sắc chỉ định, nét vẽ mỏng hơn."""
    for i in range(1, len(path)):
        if path[i-1][0] == path[i][0] or path[i-1][1] == path[i][1]:
            pygame.draw.line(window, color, path[i-1], path[i], 2)  # Độ dày 2 pixel cho ngang/dọc
        else:
            pygame.draw.line(window, color, path[i-1], path[i], 3)  # Độ dày 3 pixel cho chéo

def draw_grid(window, cell_size, env_size, env_padding):
    """Vẽ lưới nền."""
    for i in range(1, int(env_size / cell_size)):
        pygame.draw.line(window, BLACK, (env_padding + i * cell_size, env_padding),
                         (env_padding + i * cell_size, env_padding + env_size), 1)
        pygame.draw.line(window, BLACK, (env_padding, env_padding + i * cell_size),
                         (env_padding + env_size, env_padding + i * cell_size), 1)

def select_map():
    """Chọn bản đồ từ người dùng."""
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

def find_latest_file(map_name, controller_name):
    """Tìm file mới nhất trong thư mục results/{map_name}/{controller_name}/ bắt đầu bằng robot_paths_."""
    results_dir = f"results/{map_name}/{controller_name}"
    if not os.path.exists(results_dir):
        return None
    
    files = [f for f in os.listdir(results_dir) if f.startswith("robot_paths_") and f.endswith(".txt")]
    if not files:
        return None
    
    def get_timestamp(file_name):
        parts = file_name.split('_')
        if len(parts) >= 4:
            timestamp_str = parts[2] + '_' + parts[3].split('.')[0]
            try:
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                return datetime.min
        return datetime.min
    
    latest_file = max(files, key=get_timestamp)
    return os.path.join(results_dir, latest_file)

def read_last_path(file_path, cell_size, env_padding, goal_grid):
    """Đọc episode đầu tiên thỏa mãn điều kiện: phần tử cuối (sau khi loại bỏ reset) có khoảng cách Euclidean <= 1 đến goal_grid. Nếu không tìm thấy, trả về episode cuối cùng."""
    episodes = []
    current_episode = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Episode"):
                if current_episode and len(current_episode) > 1:  # Chỉ lưu episode nếu có hơn 1 điểm
                    episodes.append(current_episode[:-1])  # Loại bỏ điểm reset ngay khi lưu episode
                current_episode = []
            elif line:
                try:
                    if line.endswith(','):
                        line = line[:-1]
                    coords_list = eval(f"[{line}]")
                    current_episode.extend(coords_list)
                except SyntaxError:
                    print(f"Error parsing line: {line}")
                    continue
    
    if current_episode and len(current_episode) > 1:  # Kiểm tra và lưu episode cuối nếu hợp lệ
        episodes.append(current_episode[:-1])  # Loại bỏ điểm reset
    
    if not episodes:
        return []
    
    # print(goal_grid)
    
    # Tìm episode đầu tiên thỏa mãn điều kiện trên tọa độ lưới
    for episode in episodes:
        if len(episode) >= 1:  # Chỉ cần 1 phần tử để lấy phần tử cuối (do đã xóa điểm reset)
            last_point = episode[-1]  # Lấy phần tử cuối (trước đây là áp chót)
            x, y = last_point
            a, b = goal_grid
            # Kiểm tra khoảng cách Euclidean trên tọa độ lưới
            if (x - a) ** 2 + (y - b) ** 2 <= 2:
                # Chuyển đổi episode sang tọa độ pixel
                pixel_path = [(env_padding + (x + 0.5) * cell_size, env_padding + (y + 0.5) * cell_size) for x, y in episode]
                return pixel_path
    
    # Nếu không tìm thấy episode nào thỏa mãn, trả về episode cuối cùng
    last_episode = episodes[-1]
    pixel_path = [(env_padding + (x + 0.5) * cell_size, env_padding + (y + 0.5) * cell_size) for x, y in last_episode]
    return pixel_path

def get_controller_color(index):
    """Trả về màu cố định cho controller dựa trên chỉ số."""
    colors = [
        (255, 0, 0),    # Đỏ
        (0, 0, 139),    # Xanh dương đậm
        (0, 255, 0),    # Xanh lá cây đậm
        (255, 215, 0),  # Vàng đậm
        (0, 0, 0),      # Đen
        (139, 69, 19)   # Nâu
    ]
    return colors[index % len(colors)]  # Lặp lại màu nếu có nhiều controller hơn số màu

def save_visualization(screen, map_name):
    """Lưu ảnh chụp màn hình Pygame vào thư mục results/<map_name>/visualization/."""
    output_dir = f"results/{map_name}/visualization"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/visualization_{map_name}_{timestamp}.png"
    pygame.image.save(screen, filename)
    print(f"Đã lưu ảnh trực quan hóa tại: {filename}")
    return filename

if __name__ == "__main__":
    # Thiết lập thông số môi trường
    cell_size = 16
    env_size = 512
    env_padding = int(env_size * 0.06)
    
    # Chọn bản đồ
    selected_map = select_map()
    
    # Tạo thư mục lưu ảnh
    os.makedirs(f"results/{selected_map}", exist_ok=True)
    
    # Tìm tất cả các controller trong thư mục results/<map_name>/
    results_dir = f"results/{selected_map}"
    if not os.path.exists(results_dir):
        print(f"No results found for map {selected_map} in results/{selected_map}/")
        exit()
    
    controllers = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not controllers:
        print(f"No controllers found in results/{selected_map}/")
        exit()
    
    # Lấy thông tin bản đồ
    start_grid = maps[selected_map]["Start"]
    goal_grid = maps[selected_map]["Goal"]
    start = (env_padding + (start_grid[0] + 0.5) * cell_size, env_padding + (start_grid[1] + 0.5) * cell_size)
    goal = (env_padding + (goal_grid[0] + 0.5) * cell_size, env_padding + (goal_grid[1] + 0.5) * cell_size)
    
    # Đọc đường đi từ mỗi controller
    paths = {}
    colors = {}
    for i, controller in enumerate(controllers):
        latest_file = find_latest_file(selected_map, controller)
        if latest_file:
            path = read_last_path(latest_file, cell_size, env_padding, goal_grid)
            if path:
                paths[controller] = path
                colors[controller] = get_controller_color(i)  # Gán màu cố định
    
    if not paths:
        print("No valid paths found for any controller.")
        exit()
    
    # Thiết lập Pygame với cửa sổ mở rộng
    pygame.init()
    screen = pygame.display.set_mode((env_size + 3 * env_padding, env_size + 3 * env_padding))  # Mở rộng chiều rộng
    pygame.display.set_caption(f"Path Visualization - All Controllers - Map: {selected_map}")
    
    # Khởi tạo chướng ngại vật
    obstacles_list = []
    if selected_map in maps:
        for obs_data in maps[selected_map]["Obstacles"]:
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
            obstacles_list.append(obstacle)
    
    running = True
    pause = False
    saved_image = False  # Cờ để đảm bảo chỉ lưu ảnh một lần
    
    while running:
        screen.fill(WHITE)
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == MOUSEBUTTONDOWN:
                if button_pause.collidepoint(event.pos):
                    pause = not pause
        
        # Vẽ nút Pause
        button_pause = pygame.draw.rect(screen, BLACK, (env_padding + int(env_size * 0.4), env_padding * 2 + env_size,
                                                       int(env_size * 0.2), int(env_padding * 0.4)), 4)
        button_pause_text = pygame.font.SysFont("arial", env_padding // 2).render("Pause", True, BLACK)
        screen.blit(button_pause_text, button_pause_text.get_rect(center=button_pause.center))
        
        if not pause:
            for obstacle in obstacles_list:
                obstacle.move()
        
        # Vẽ lưới
        draw_grid(screen, cell_size, env_size, env_padding)
        
        # Vẽ chướng ngại vật
        for obstacle in obstacles_list:
            obstacle.draw(screen)
        
        # Vẽ tất cả các đường đi
        for controller, path in paths.items():
            draw_path(screen, path, colors[controller])
        
        # Vẽ điểm bắt đầu và mục tiêu
        draw_start(screen, start)
        draw_target(screen, goal)
        
        # Vẽ viền
        pygame.draw.rect(screen, BLACK, (env_padding, env_padding, env_size, env_size), 3)
        
        # Vẽ chú thích ở góc trên bên phải, căn trái, cỡ chữ nhỏ hơn, nền trắng
        font = pygame.font.SysFont("arial", 12)  # Cỡ chữ 12
        x_offset = env_size + env_padding + 10  # Cách lưới 10 pixel bên phải
        y_offset = 50  # Cách mép trên 50 pixel
        # Tính kích thước nền trắng
        max_width = max([font.render(controller, True, colors[controller]).get_width() for controller in paths.keys()] + [0])
        legend_height = len(paths) * 17  # Mỗi dòng cao 17 pixel
        legend_rect = pygame.Rect(x_offset - 5, y_offset - 5, max_width + 10, legend_height + 10)
        pygame.draw.rect(screen, WHITE, legend_rect)  # Vẽ nền trắng
        pygame.draw.rect(screen, BLACK, legend_rect, 1)  # Vẽ viền đen cho nền
        for i, controller in enumerate(paths.keys()):
            text = font.render(controller, True, colors[controller])
            text_rect = text.get_rect(topleft=(x_offset, y_offset + i * 17))  # Căn trái, cách 17 pixel
            screen.blit(text, text_rect)
        
        pygame.display.update()
        
        # Lưu ảnh sau lần vẽ đầu tiên
        if not saved_image:
            save_visualization(screen, selected_map)
            saved_image = True
    
    pygame.quit()