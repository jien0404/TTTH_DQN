import pygame

class GridEnv:
    def __init__(self, grid_size=20, cell_size=30):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
        self.clock = pygame.time.Clock()

        self.robot_pos = [10, 10]
        self.trail = []

    def draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

    def draw_robot(self):
        x, y = self.robot_pos
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, (0, 255, 0), (center_x, center_y), self.cell_size // 3)

    def draw_trail(self):
        for pos in self.trail:
            x, y = pos
            rect = pygame.Rect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 165, 0), rect)

    def draw_vision(self):
        x, y = self.robot_pos
        for i in range(-2, 3):
            for j in range(-2, 3):
                nx, ny = x + i, y + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    rect = pygame.Rect(nx*self.cell_size, ny*self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (173, 216, 230), rect, 2)

    def move_robot(self, dx, dy):
        nx = self.robot_pos[0] + dx
        ny = self.robot_pos[1] + dy
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            self.trail.append(tuple(self.robot_pos))
            self.robot_pos = [nx, ny]

    def step(self):
        self.screen.fill((255, 255, 255))
        self.draw_trail()
        self.draw_grid()
        self.draw_vision()
        self.draw_robot()
        pygame.display.flip()
        self.clock.tick(10)
