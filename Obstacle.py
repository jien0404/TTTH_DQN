import pygame
import numpy as np
from Colors import *

class Obstacle:
    def __init__(self, x, y, width, height, static, velocity=None, x_bound=None, y_bound=None, path=None, angle=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.static = static
        self.angle = angle  # Góc xoay (độ)
        self.velocity = np.array(velocity if velocity else [0, 0])
        self.x_bound = x_bound if x_bound else (x - width, x + width)
        self.y_bound = y_bound if y_bound else (y - height, y + height)
        self.path = path if path else []
        self.path_index = 0
        self.history = []  # Lưu lịch sử vị trí để vẽ bóng mờ (nếu cần)

    def move(self):
        if self.static:
            return

        self.history.append((self.x, self.y))
        if len(self.history) > 50:  # Giới hạn lịch sử
            self.history.pop(0)

        # Di chuyển theo chu kỳ (path)
        if self.path:
            if self.path_index < len(self.path):
                target_x, target_y = self.path[self.path_index]
                dx = target_x - self.x
                dy = target_y - self.y
                norm = (dx**2 + dy**2)**0.5
                if norm < 1:  # Đến gần điểm đích
                    self.path_index = (self.path_index + 1) % len(self.path)
                else:
                    speed = 1.5
                    self.x += (dx/norm) * speed
                    self.y += (dy/norm) * speed
        else:
            # Di chuyển dao động
            self.x += self.velocity[0]
            self.y += self.velocity[1]
            if self.x < self.x_bound[0]:
                self.x = self.x_bound[0]
                self.velocity[0] = -self.velocity[0]
            elif self.x > self.x_bound[1]:
                self.x = self.x_bound[1]
                self.velocity[0] = -self.velocity[0]
            if self.y < self.y_bound[0]:
                self.y = self.y_bound[0]
                self.velocity[1] = -self.velocity[1]
            elif self.y > self.y_bound[1]:
                self.y = self.y_bound[1]
                self.velocity[1] = -self.velocity[1]

    def draw(self, window):
        # Tạo surface cho chướng ngại vật
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        color = BLACK if self.static else CYAN
        pygame.draw.rect(surface, color, (0, 0, self.width, self.height))
        
        # Xoay surface nếu có góc
        if self.angle != 0:
            surface = pygame.transform.rotate(surface, self.angle)
        
        # Tính vị trí để vẽ
        rect = surface.get_rect(center=(self.x, self.y))
        window.blit(surface, rect.topleft)

    def get_bounding_box(self):
        # Trả về tọa độ bao quanh (dùng để kiểm tra va chạm)
        # Điều chỉnh tọa độ dựa trên góc xoay
        half_w, half_h = self.width / 2, self.height / 2
        return (self.x - half_w, self.x + half_w, self.y - half_h, self.y + half_h)