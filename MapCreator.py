import pygame
import numpy as np
from pygame.locals import *
import os
import sys
import uuid

# Thêm đường dẫn đến thư mục environment
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/environment')
try:
    from Colors import *
    from Obstacle import Obstacle
except ImportError as e:
    print(f"Error importing environment modules: {e}")
    sys.exit(1)

class MapCreator:
    def __init__(self, cell_size=16, env_size=512, env_padding=None):
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
        pygame.display.set_caption("Map Creator - Draw Obstacles")
        self.my_font = pygame.font.SysFont("arial", self.SOUTH_PAD // 5)

        # Biến trạng thái
        self.drawing = False
        self.is_static = True
        self.obstacles = []
        self.start_pos = None
        self.new_pos = None
        self.map_name = f"map{len(self.load_existing_maps()) + 1}"
        self.angle = 0  # Góc xoay mặc định
        self.dragging_slider = False

        # Thiết lập nút và thanh trượt giao diện
        self.setup_buttons()

    def setup_buttons(self):
        """Thiết lập các nút và thanh trượt giao diện"""
        button_width = int(self.env_width * 0.15)
        button_height = int(self.SOUTH_PAD * 0.4)
        button_y = self.NORTH_PAD * 2 + self.env_height

        self.button_save = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.01), button_y, button_width, button_height)
        self.button_static_dynamic = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.17), button_y, button_width, button_height)
        self.button_undo = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.33), button_y, button_width, button_height)
        self.button_clear = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.49), button_y, button_width, button_height)

        # Thanh trượt cho góc xoay
        self.slider_rect = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.65), button_y, int(self.env_width * 0.3), button_height)
        self.slider_handle = pygame.Rect(self.LEFT_PAD + int(self.env_width * 0.65), button_y, 10, button_height)

        self.button_texts = {
            'save': self.my_font.render("Save", True, BLACK),
            'static_dynamic': self.my_font.render("Static" if self.is_static else "Dynamic", True, BLACK),
            'undo': self.my_font.render("Undo", True, BLACK),
            'clear': self.my_font.render("Clear All", True, BLACK),
            'angle': self.my_font.render(f"Angle: {self.angle}°", True, BLACK)
        }

    def load_existing_maps(self):
        """Tải danh sách các bản đồ hiện có từ MapData.py"""
        try:
            # Import MapData.py từ cùng thư mục
            sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
            from MapData import maps
            return maps
        except ImportError:
            return {}  # Trả về dictionary rỗng nếu MapData.py không tồn tại

    def draw_grid(self):
        """Vẽ lưới nền"""
        self.screen.fill(WHITE)
        for i in range(1, int(self.env_size / self.cell_size)):
            x = self.LEFT_PAD + i * self.cell_size
            y = self.NORTH_PAD + i * self.cell_size
            pygame.draw.line(self.screen, BLACK, (x, self.NORTH_PAD), (x, self.NORTH_PAD + self.env_size), 1)
            pygame.draw.line(self.screen, BLACK, (self.LEFT_PAD, y), (self.LEFT_PAD + self.env_size, y), 1)

    def draw_buttons(self):
        """Vẽ các nút và thanh trượt giao diện"""
        # Vẽ các nút
        for button in [self.button_save, self.button_static_dynamic, self.button_undo, self.button_clear]:
            pygame.draw.rect(self.screen, BLACK, button, 4)
        self.screen.blit(self.button_texts['save'], self.button_texts['save'].get_rect(center=self.button_save.center))
        self.screen.blit(self.button_texts['static_dynamic'], self.button_texts['static_dynamic'].get_rect(center=self.button_static_dynamic.center))
        self.screen.blit(self.button_texts['undo'], self.button_texts['undo'].get_rect(center=self.button_undo.center))
        self.screen.blit(self.button_texts['clear'], self.button_texts['clear'].get_rect(center=self.button_clear.center))

        # Vẽ thanh trượt
        pygame.draw.rect(self.screen, BLACK, self.slider_rect, 2)
        pygame.draw.rect(self.screen, GRAY, self.slider_handle)
        self.button_texts['angle'] = self.my_font.render(f"Angle: {self.angle}°", True, BLACK)
        self.screen.blit(self.button_texts['angle'], (self.slider_rect.x + 10, self.slider_rect.y - self.SOUTH_PAD // 4))

    def update_slider(self, mouse_x):
        """Cập nhật vị trí thanh trượt và tính góc xoay"""
        slider_width = self.slider_rect.width - self.slider_handle.width
        relative_x = max(0, min(mouse_x - self.slider_rect.x, slider_width))
        self.slider_handle.x = self.slider_rect.x + relative_x
        self.angle = int((relative_x / slider_width) * 360)  # Chuyển đổi vị trí thành góc từ 0 đến 360

    def save_map(self):
        """Lưu bản đồ vào MapData.py"""
        map_data = {
            "Start": (1, 1),  # Mặc định điểm bắt đầu
            "Goal": (30, 30),  # Mặc định điểm kết thúc
            "Obstacles": []
        }

        for obstacle in self.obstacles:
            obs_data = {
                "x": obstacle.x,
                "y": obstacle.y,
                "width": obstacle.width,
                "height": obstacle.height,
                "static": obstacle.static,
                "angle": obstacle.angle
            }
            if not obstacle.static:
                obs_data["velocity"] = obstacle.velocity.tolist()
                obs_data["x_bound"] = obstacle.x_bound
                obs_data["y_bound"] = obstacle.y_bound
                if obstacle.path:
                    obs_data["path"] = obstacle.path
            map_data["Obstacles"].append(obs_data)

        # Đường dẫn đến file MapData.py (cùng thư mục với MapCreator.py)
        map_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MapData.py')

        # Tải các bản đồ hiện có
        existing_maps = self.load_existing_maps()
        existing_maps[self.map_name] = map_data

        # Ghi vào file MapData.py
        try:
            with open(map_file_path, 'w') as f:
                f.write("# Start và Goal được định nghĩa bằng chỉ số ô lưới (grid_x, grid_y)\n")
                f.write("maps = {\n")
                for name, data in existing_maps.items():
                    f.write(f"    \"{name}\": {{\n")
                    f.write(f"        \"Start\": {data['Start']},\n")
                    f.write(f"        \"Goal\": {data['Goal']},\n")
                    f.write(f"        \"Obstacles\": [\n")
                    for obs in data["Obstacles"]:
                        f.write("            {\n")
                        for key, value in obs.items():
                            f.write(f"                \"{key}\": {value},\n")
                        f.write("            },\n")
                    f.write("        ]\n")
                    f.write("    },\n")
                f.write("}\n")
            print(f"Map saved as {self.map_name} in MapData.py")
        except Exception as e:
            print(f"Error saving MapData.py: {e}")

    def run(self):
        """Chạy vòng lặp chính của MapCreator"""
        running = True
        while running:
            self.draw_grid()
            self.draw_buttons()

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

                if event.type == MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if self.button_save.collidepoint(mouse_x, mouse_y):
                        self.save_map()
                        running = False
                    elif self.button_static_dynamic.collidepoint(mouse_x, mouse_y):
                        self.is_static = not self.is_static
                        self.button_texts['static_dynamic'] = self.my_font.render("Static" if self.is_static else "Dynamic", True, BLACK)
                        print("Mode:", "Static" if self.is_static else "Dynamic")
                    elif self.button_undo.collidepoint(mouse_x, mouse_y):
                        if self.obstacles:
                            self.obstacles.pop()
                            print("Last obstacle removed")
                    elif self.button_clear.collidepoint(mouse_x, mouse_y):
                        self.obstacles.clear()
                        print("All obstacles cleared")
                    elif self.slider_rect.collidepoint(mouse_x, mouse_y):
                        self.dragging_slider = True
                        self.update_slider(mouse_x)
                    elif self.LEFT_PAD <= mouse_x <= self.LEFT_PAD + self.env_size and \
                         self.NORTH_PAD <= mouse_y <= self.NORTH_PAD + self.env_size:
                        self.start_pos = (mouse_x, mouse_y)
                        self.drawing = True

                if event.type == MOUSEMOTION and self.dragging_slider:
                    self.update_slider(event.pos[0])

                if event.type == MOUSEBUTTONUP:
                    if self.dragging_slider:
                        self.dragging_slider = False
                    elif self.drawing:
                        self.new_pos = event.pos
                        width = abs(self.new_pos[0] - self.start_pos[0])
                        height = abs(self.new_pos[1] - self.start_pos[1])
                        if width > 0 and height > 0:  # Đảm bảo kích thước hợp lệ
                            x = (self.start_pos[0] + self.new_pos[0]) / 2
                            y = (self.start_pos[1] + self.new_pos[1]) / 2
                            velocity = [float(np.random.randn() * 2), 0] if not self.is_static else None
                            x_bound = (x - width, x + width) if not self.is_static else None
                            y_bound = (y - height, y - height) if not self.is_static else None
                            new_obstacle = Obstacle(
                                x=x,
                                y=y,
                                width=width,
                                height=height,
                                static=self.is_static,
                                velocity=velocity,
                                x_bound=x_bound,
                                y_bound=y_bound,
                                angle=self.angle  # Sử dụng góc từ thanh trượt
                            )
                            self.obstacles.append(new_obstacle)
                            print(f"Added {'static' if self.is_static else 'dynamic'} obstacle at ({x}, {y}) with angle {self.angle}°")
                        self.drawing = False

                if event.type == KEYDOWN:
                    if event.key == K_z and pygame.key.get_mods() & KMOD_CTRL:
                        if self.obstacles:
                            self.obstacles.pop()
                            print("Last obstacle removed")

            if self.drawing and self.start_pos:
                new_mx, new_my = pygame.mouse.get_pos()
                # Tạo surface tạm thời để vẽ chướng ngại vật đang kéo
                surface = pygame.Surface((abs(new_mx - self.start_pos[0]), abs(new_my - self.start_pos[1])), pygame.SRCALPHA)
                color = BLACK if self.is_static else CYAN
                pygame.draw.rect(surface, color, (0, 0, abs(new_mx - self.start_pos[0]), abs(new_my - self.start_pos[1])))
                if self.angle != 0:
                    surface = pygame.transform.rotate(surface, self.angle)
                rect = surface.get_rect(center=((self.start_pos[0] + new_mx) / 2, (self.start_pos[1] + new_my) / 2))
                self.screen.blit(surface, rect.topleft)

            for obstacle in self.obstacles:
                obstacle.draw(self.screen)

            pygame.draw.rect(self.screen, BLACK, (self.LEFT_PAD, self.NORTH_PAD, self.env_width, self.env_height), 3)
            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    creator = MapCreator()
    creator.run()