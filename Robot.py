import pygame
import numpy as np
from Colors import *

class Robot:
    def __init__(self, x, y, cell_size, controller, vision, radius=8, env_padding=30):
        self.cell_size = cell_size
        self.env_padding = env_padding
        self.grid_x = int((x - env_padding) // cell_size)
        self.grid_y = int((y - env_padding) // cell_size)
        self.x = self.env_padding + (self.grid_x + 0.5) * self.cell_size
        self.y = self.env_padding + (self.grid_y + 0.5) * self.cell_size
        self.controller = controller
        self.vision = vision
        self.radius = radius
        self.direction = (0, 0)

    def move(self, obstacles, grid_width, grid_height):
        self.direction = self.controller.make_decision(self, obstacles)
        dx, dy = self.direction
        new_grid_x = self.grid_x + dx
        new_grid_y = self.grid_y + dy

        # Kiểm tra xem vị trí mới có nằm trong lưới hay không
        if not (0 <= new_grid_x < grid_width and 0 <= new_grid_y < grid_height):
            # Nếu vượt biên, coi như va chạm với chướng ngại vật
            return True  # Trả về True để báo hiệu va chạm

        # Kiểm tra diagonal squeeze - chặn di chuyển qua khe chéo
        if self.is_diagonal_move(dx, dy):
            if self.check_diagonal_squeeze(obstacles, dx, dy, grid_width, grid_height):
                return True  # Chặn di chuyển qua khe chéo

        # Lưu vị trí cũ để khôi phục nếu va chạm
        old_grid_x, old_grid_y = self.grid_x, self.grid_y
        old_x, old_y = self.x, self.y

        # Cập nhật tạm thời vị trí
        self.grid_x = new_grid_x
        self.grid_y = new_grid_y
        self.x = self.env_padding + (self.grid_x + 0.5) * self.cell_size
        self.y = self.env_padding + (self.grid_y + 0.5) * self.cell_size

        # Kiểm tra va chạm với chướng ngại vật
        if self.check_collision(obstacles):
            # Nếu va chạm, khôi phục vị trí cũ
            # self.grid_x, self.grid_y = old_grid_x, old_grid_y
            # self.x, self.y = old_x, old_y
            return True  # Báo hiệu va chạm

        return False  # Không có va chạm

    def is_diagonal_move(self, dx, dy):
        """Kiểm tra xem di chuyển có phải là diagonal không"""
        return abs(dx) == 1 and abs(dy) == 1

    def check_diagonal_squeeze(self, obstacles, dx, dy, grid_width, grid_height):
        """
        Kiểm tra xem di chuyển diagonal có bị chặn bởi 2 chướng ngại vật chéo không
        
        Ví dụ: Robot ở (1,0) muốn đi đến (0,1)
        Kiểm tra xem có chướng ngại vật ở (0,0) VÀ (1,1) không
        Nếu có thì chặn di chuyển
        """
        current_x, current_y = self.grid_x, self.grid_y
        
        # 2 ô cần kiểm tra để tránh diagonal squeeze
        corner1_x = current_x + dx  # Ví dụ: (1,0) -> (0,1) thì corner1 = (0,0) 
        corner1_y = current_y       # dx=-1, dy=+1 -> corner1 = (1-1, 0) = (0,0)
        
        corner2_x = current_x       # corner2 = (1,1)
        corner2_y = current_y + dy  # corner2 = (1, 0+1) = (1,1)
        
        # Kiểm tra xem cả 2 corner có nằm trong grid không
        if not (0 <= corner1_x < grid_width and 0 <= corner1_y < grid_height):
            return False
        if not (0 <= corner2_x < grid_width and 0 <= corner2_y < grid_height):
            return False
            
        # Kiểm tra xem có chướng ngại vật ở cả 2 corner không
        corner1_blocked = self.is_grid_cell_blocked(obstacles, corner1_x, corner1_y)
        corner2_blocked = self.is_grid_cell_blocked(obstacles, corner2_x, corner2_y)
        
        # Nếu cả 2 corner đều có chướng ngại vật thì chặn di chuyển
        return corner1_blocked and corner2_blocked

    def is_grid_cell_blocked(self, obstacles, grid_x, grid_y):
        """Kiểm tra xem một ô grid có bị chặn bởi chướng ngại vật không"""
        # Chuyển đổi grid coordinates sang pixel coordinates
        cell_center_x = self.env_padding + (grid_x + 0.5) * self.cell_size
        cell_center_y = self.env_padding + (grid_y + 0.5) * self.cell_size
        
        for obs in obstacles:
            x1, x2, y1, y2 = obs.get_bounding_box()
            # Kiểm tra xem center của cell có nằm trong bounding box của obstacle không
            if x1 <= cell_center_x <= x2 and y1 <= cell_center_y <= y2:
                return True
        return False

    def draw(self, window):
        pygame.draw.circle(window, RED, (int(self.x), int(self.y)), self.vision, 1)
        pygame.draw.circle(window, RED, (int(self.x), int(self.y)), self.radius)

    def check_collision(self, obstacles):
        for obs in obstacles:
            x1, x2, y1, y2 = obs.get_bounding_box()
            closest_x = max(x1, min(self.x, x2))
            closest_y = max(y1, min(self.y, y2))
            distance = ((closest_x - self.x)**2 + (closest_y - self.y)**2) ** 0.5
            if distance < self.radius:
                return True
        return False

    def get_state(self, obstacles, grid_width, grid_height, goal):
        # Tạo ma trận trạng thái 5x5, khởi tạo tất cả là 0 (ô trống)
        state = np.zeros((5, 5), dtype=int)

        # Duyệt các ô trong phạm vi 5x5 xung quanh robot
        for i in range(-2, 3):
            for j in range(-2, 3):
                grid_x = self.grid_x + i
                grid_y = self.grid_y + j
                state_idx_x = i + 2
                state_idx_y = j + 2
                if not (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
                    state[state_idx_x, state_idx_y] = 1
                    continue

        # Duyệt các chướng ngại vật và gán giá trị 1 cho các ô mà chúng chiếm
        for obs in obstacles:
            x1, x2, y1, y2 = obs.get_bounding_box()
            grid_x1 = int((x1 - self.env_padding) // self.cell_size)
            grid_x2 = int((x2 - self.env_padding) // self.cell_size)
            grid_y1 = int((y1 - self.env_padding) // self.cell_size)
            grid_y2 = int((y2 - self.env_padding) // self.cell_size)
            for gx in range(max(0, grid_x1), min(grid_width, grid_x2 + 1)):
                for gy in range(max(0, grid_y1), min(grid_height, grid_y2 + 1)):
                    state_idx_x = gx - self.grid_x + 2
                    state_idx_y = gy - self.grid_y + 2
                    if 0 <= state_idx_x < 5 and 0 <= state_idx_y < 5:
                        state[state_idx_x, state_idx_y] = 1

        # Gán giá trị 2 cho vị trí robot (trung tâm ma trận)
        state[2, 2] = 2

        # Tính chỉ số ô lưới của đích
        goal_grid_x = int((goal[0] - self.env_padding) // self.cell_size)
        goal_grid_y = int((goal[1] - self.env_padding) // self.cell_size)

        # Tính khoảng cách Euclidean đến đích (tính bằng số ô)
        distance_to_goal = ((self.grid_x - goal_grid_x)**2 + (self.grid_y - goal_grid_y)**2)**0.5

        return state, distance_to_goal
    
    def distance_to_nearest_obstacle(self, obstacles):
        """Calculate the distance to the nearest obstacle"""
        min_distance = float('inf')

        for obs in obstacles:
            x1, x2, y1, y2 = obs.get_bounding_box()

            # Find the closest point on the obstacle to the robot
            closest_x = max(x1, min(self.x, x2))
            closest_y = max(y1, min(self.y, y2))

            # Calculate Euclidean distance from robot to this closest point
            distance = ((closest_x - self.x)**2 + (closest_y - self.y)**2) ** 0.5

            # Update minimum distance if this obstacle is closer
            if distance < min_distance:
                min_distance = distance

        return min_distance
    
    def detect_moving_obstacles(self, obstacles):
        """
        Detect moving obstacles within the robot's vision range
        Returns a list of moving obstacles and their distances
        """
        moving_obstacles = []
        
        for obs in obstacles:
            # Skip static obstacles
            if obs.static:
                continue
                
            # Calculate the center of the obstacle
            obs_center_x = obs.x
            obs_center_y = obs.y
            
            # Calculate distance to the obstacle's center
            distance = ((obs_center_x - self.x)**2 + (obs_center_y - self.y)**2) ** 0.5
            
            # Check if obstacle is within vision range
            if distance <= self.vision:
                # Store the obstacle and its distance
                moving_obstacles.append({
                    "obstacle": obs,
                    "distance": distance,
                    "direction": (obs_center_x - self.x, obs_center_y - self.y),
                    "velocity": obs.velocity
                })
        
        # Sort by distance (closest first)
        moving_obstacles.sort(key=lambda x: x["distance"])
        
        return moving_obstacles