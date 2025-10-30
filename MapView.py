import pygame
from pygame.locals import *
import os
import sys

# Thêm đường dẫn đến thư mục environment
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/environment')
from Colors import *
from Obstacle import Obstacle
from MapData import maps

def select_map():
    """Hàm chọn bản đồ từ danh sách bản đồ trong MapData.py"""
    print("Available maps:")
    for i, map_name in enumerate(maps.keys()):
        print(f"{i+1}. {map_name}")
    
    while True:
        try:
            choice = input(f"Select map (1-{len(maps)}): ")
            if choice == "":
                return list(maps.keys())[0]  # Chọn bản đồ đầu tiên nếu không nhập
            choice = int(choice)
            if 1 <= choice <= len(maps):
                return list(maps.keys())[choice-1]
            print(f"Please enter a number between 1 and {len(maps)}")
        except ValueError:
            print("Please enter a valid number or press Enter for default (map1)")

class MapView:
    def __init__(self, cell_size=16, env_size=512, env_padding=None, selected_map=None):
        # Thiết lập thông số môi trường
        self.cell_size = cell_size
        self.env_size = env_size
        self.env_padding = int(env_size * 0.06) if env_padding is None else env_padding
        self.env_width = self.env_height = env_size

        # Thiết lập cửa sổ Pygame
        self.NORTH_PAD, self.SOUTH_PAD, self.LEFT_PAD, self.RIGHT_PAD = self.env_padding, 3 * self.env_padding, self.env_padding, self.env_padding
        self.SCREEN_WIDTH = self.env_width + self.LEFT_PAD + self.RIGHT_PAD
        self.SCREEN_HEIGHT = self.env_height + self.NORTH_PAD + self.SOUTH_PAD
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption("Map Viewer")
        self.my_font = pygame.font.SysFont("arial", self.SOUTH_PAD // 5)

        # Danh sách bản đồ và trạng thái
        self.maps = maps
        self.map_names = list(maps.keys())
        self.current_map_index = 0
        self.current_map = selected_map if selected_map in self.map_names else self.map_names[0] if self.map_names else None
        self.obstacles = []
        self.start = None
        self.goal = None

        # Thiết lập nút giao diện
        self.setup_buttons()

        # Tải bản đồ
        if self.current_map:
            self.load_map(self.current_map)

    def setup_buttons(self):
        """Thiết lập các nút giao diện"""
        button_width = int(self.env_width * 0.15)
        button_height = int(self.SOUTH_PAD * 0.4)
        button_y = self.NORTH_PAD * 2 + self.env_height

        self.button_prev = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.01), button_y, button_width, button_height)
        self.button_next = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.17), button_y, button_width, button_height)
        self.button_exit = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.81), button_y, button_width, button_height)

        self.button_texts = {
            'prev': self.my_font.render("Prev Map", True, BLACK),
            'next': self.my_font.render("Next Map", True, BLACK),
            'exit': self.my_font.render("Exit", True, BLACK),
            'map_name': self.my_font.render(f"Map: {self.current_map}", True, BLACK)
        }

    def load_map(self, map_name):
        """Tải dữ liệu bản đồ từ MapData.py"""
        if map_name not in self.maps:
            print(f"Map {map_name} not found!")
            return

        self.current_map = map_name
        self.current_map_index = self.map_names.index(map_name)
        self.button_texts['map_name'] = self.my_font.render(f"Map: {self.current_map}", True, BLACK)
        map_data = self.maps[map_name]

        # Tải điểm bắt đầu và kết thúc
        start_grid_x, start_grid_y = map_data["Start"]
        goal_grid_x, goal_grid_y = map_data["Goal"]
        self.start = (
            self.env_padding + (start_grid_x + 0.5) * self.cell_size,
            self.env_padding + (start_grid_y + 0.5) * self.cell_size
        )
        self.goal = (
            self.env_padding + (goal_grid_x + 0.5) * self.cell_size,
            self.env_padding + (goal_grid_y + 0.5) * self.cell_size
        )

        # Tải chướng ngại vật
        self.obstacles = []
        for obs_data in map_data["Obstacles"]:
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

    def draw_grid(self):
        """Vẽ lưới nền"""
        self.screen.fill(WHITE)
        for i in range(1, int(self.env_size / self.cell_size)):
            x = self.LEFT_PAD + i * self.cell_size
            y = self.NORTH_PAD + i * self.cell_size
            pygame.draw.line(self.screen, BLACK, (x, self.NORTH_PAD), (x, self.NORTH_PAD + self.env_size), 1)
            pygame.draw.line(self.screen, BLACK, (self.LEFT_PAD, y), (self.LEFT_PAD + self.env_size, y), 1)

    def draw_environment(self):
        """Vẽ toàn bộ môi trường"""
        self.draw_grid()

        # Vẽ chướng ngại vật
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        # Vẽ điểm bắt đầu và kết thúc
        if self.start:
            pygame.draw.rect(self.screen, GREEN, (self.start[0] - 5, self.start[1] - 5, 10, 10))
        if self.goal:
            pygame.draw.circle(self.screen, RED, self.goal, 8)

        # Vẽ viền môi trường
        pygame.draw.rect(self.screen, BLACK, (self.LEFT_PAD, self.NORTH_PAD, self.env_width, self.env_height), 3)

        # Vẽ các nút
        pygame.draw.rect(self.screen, BLACK, self.button_prev, 4)
        pygame.draw.rect(self.screen, BLACK, self.button_next, 4)
        pygame.draw.rect(self.screen, BLACK, self.button_exit, 4)
        self.screen.blit(self.button_texts['prev'], self.button_texts['prev'].get_rect(center=self.button_prev.center))
        self.screen.blit(self.button_texts['next'], self.button_texts['next'].get_rect(center=self.button_next.center))
        self.screen.blit(self.button_texts['exit'], self.button_texts['exit'].get_rect(center=self.button_exit.center))
        self.screen.blit(self.button_texts['map_name'], (self.LEFT_PAD + int(self.env_width * 0.33), self.NORTH_PAD * 2 + self.env_height))

    def run(self):
        """Chạy vòng lặp chính của MapView"""
        if not self.map_names:
            print("No maps available in MapData.py!")
            pygame.quit()
            sys.exit()

        running = True
        clock = pygame.time.Clock()
        FPS = 60

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if self.button_prev.collidepoint(mouse_x, mouse_y):
                        self.current_map_index = (self.current_map_index - 1) % len(self.map_names)
                        self.load_map(self.map_names[self.current_map_index])
                    elif self.button_next.collidepoint(mouse_x, mouse_y):
                        self.current_map_index = (self.current_map_index + 1) % len(self.map_names)
                        self.load_map(self.map_names[self.current_map_index])
                    elif self.button_exit.collidepoint(mouse_x, mouse_y):
                        running = False

            # Cập nhật vị trí chướng ngại vật động
            for obstacle in self.obstacles:
                obstacle.move()

            # Vẽ môi trường
            self.draw_environment()
            pygame.display.update()
            clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    selected_map = select_map()
    viewer = MapView(selected_map=selected_map)
    viewer.run()