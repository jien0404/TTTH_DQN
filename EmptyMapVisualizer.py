import pygame
import os
import sys
import math
import numpy as np

# ======================================================================
# THIẾT LẬP ĐƯỜNG DẪN IMPORT
# ======================================================================
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'environment'))

try:
    from MapData import maps
    from Colors import *
except ImportError as e:
    print(f"Lỗi: Không tìm thấy module MapData hoặc Colors. Chi tiết: {e}")
    sys.exit(1)

# ======================================================================
# CẤU HÌNH
# ======================================================================
ENV_SIZE = 512
CELL_SIZE = 16
ENV_PADDING = int(ENV_SIZE * 0.06)
WINDOW_WIDTH = ENV_SIZE + 2 * ENV_PADDING
WINDOW_HEIGHT = ENV_SIZE + 2 * ENV_PADDING

# Màu sắc
TRAJECTORY_COLOR = (100, 149, 237)   # Cornflower Blue (Trục chính)
AREA_BORDER_COLOR = (100, 149, 237, 100) # Màu viền vùng di chuyển 2D
ARROW_COLOR = (255, 69, 0)           # Red Orange
DYNAMIC_OBS_COLOR = (70, 130, 180)   # Steel Blue

# ======================================================================
# HÀM HỖ TRỢ VẼ
# ======================================================================

def draw_dashed_rect(surface, color, rect, width=1, dash_length=10):
    """Vẽ hình chữ nhật nét đứt để thể hiện vùng di chuyển"""
    x, y, w, h = rect
    # Vẽ 4 cạnh thủ công để tạo nét đứt
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        dist = math.dist(p1, p2)
        if dist == 0: continue
        
        dx = (p2[0] - p1[0]) / dist
        dy = (p2[1] - p1[1]) / dist
        
        curr_dist = 0
        while curr_dist < dist:
            start = (p1[0] + dx * curr_dist, p1[1] + dy * curr_dist)
            end_dist = min(curr_dist + dash_length, dist)
            end = (p1[0] + dx * end_dist, p1[1] + dy * end_dist)
            pygame.draw.line(surface, color, start, end, width)
            curr_dist += dash_length * 2  # Bỏ qua khoảng trống

def draw_arrow(surface, start_pos, end_pos, color=ARROW_COLOR, width=2):
    """Vẽ mũi tên"""
    if math.dist(start_pos, end_pos) < 2: return
    pygame.draw.line(surface, color, start_pos, end_pos, width)
    
    rotation = math.atan2(start_pos[1] - end_pos[1], end_pos[0] - start_pos[0]) + math.pi/2
    rad = 6
    p1 = (end_pos[0] + math.sin(rotation + math.pi/3) * rad, end_pos[1] + math.cos(rotation + math.pi/3) * rad)
    p2 = (end_pos[0] + math.sin(rotation - math.pi/3) * rad, end_pos[1] + math.cos(rotation - math.pi/3) * rad)
    pygame.draw.polygon(surface, color, [end_pos, p1, p2])

def draw_trajectory_axis(surface, obs_data):
    """
    Phân tích logic di chuyển của Obstacle để vẽ quỹ đạo:
    1. Path: Vẽ đường nối các điểm.
    2. Bounds:
       - Nếu cả x_bound và y_bound đều có phạm vi rộng -> Di chuyển vùng (Chéo).
       - Nếu chỉ x_bound rộng -> Di chuyển Ngang.
       - Nếu chỉ y_bound rộng -> Di chuyển Dọc.
    """
    cx = int(obs_data['x']) 
    cy = int(obs_data['y'])
    
    # --- 1. Ưu tiên PATH ---
    if 'path' in obs_data and obs_data['path'] and len(obs_data['path']) > 0:
        points = [(p[0], p[1]) for p in obs_data['path']]
        if len(points) > 1:
            pygame.draw.lines(surface, TRAJECTORY_COLOR, True, points, 2) # True = Closed loop
            for p in points:
                pygame.draw.circle(surface, TRAJECTORY_COLOR, (int(p[0]), int(p[1])), 3)
                # Vẽ mũi tên chỉ hướng đi giữa các điểm
            return # Đã vẽ path thì thôi vẽ bound

    # --- 2. Xử lý BOUNDS (Di chuyển theo vận tốc & giới hạn) ---
    # Lấy bound, nếu không có trong data thì mặc định là vị trí hiện tại (không di chuyển)
    # Class Obstacle logic: x_bound=(x-w, x+w) nếu ko truyền. 
    # Nhưng ở đây ta kiểm tra data gốc từ map.
    
    x_bound = obs_data.get('x_bound')
    y_bound = obs_data.get('y_bound')

    min_x, max_x = x_bound if x_bound else (cx, cx)
    min_y, max_y = y_bound if y_bound else (cy, cy)

    range_x = abs(max_x - min_x)
    range_y = abs(max_y - min_y)
    
    # Ngưỡng để xác định có di chuyển hay không (lớn hơn 1 pixel)
    moves_x = range_x > 1.0
    moves_y = range_y > 1.0

    if moves_x and moves_y:
        # === TRƯỜNG HỢP 3: DI CHUYỂN CHÉO / VÙNG 2D ===
        # Vẽ khung bao quanh vùng di chuyển
        rect_w = range_x
        rect_h = range_y
        rect_x = min_x
        rect_y = min_y
        
        # Vẽ khung chữ nhật nét đứt
        draw_dashed_rect(surface, TRAJECTORY_COLOR, (rect_x, rect_y, rect_w, rect_h), width=2)
        
        # Vẽ mũi tên chéo thể hiện hướng di chuyển "tổng quát"
        # Vẽ từ góc này sang góc kia
        start_pt = (min_x, min_y)
        end_pt = (max_x, max_y)
        pygame.draw.line(surface, TRAJECTORY_COLOR, start_pt, end_pt, 1)
        
        # Vẽ các mũi tên chỉ hướng ở 4 góc để nhấn mạnh sự nảy
        offset = 10
        # Mũi tên đi xuống phải
        draw_arrow(surface, (min_x + offset, min_y + offset), (min_x + offset + 15, min_y + offset + 15), ARROW_COLOR)
        # Mũi tên đi lên trái
        draw_arrow(surface, (max_x - offset, max_y - offset), (max_x - offset - 15, max_y - offset - 15), ARROW_COLOR)

    elif moves_x:
        # === TRƯỜNG HỢP 1: DI CHUYỂN NGANG ===
        start_pt = (min_x, cy)
        end_pt = (max_x, cy)
        pygame.draw.line(surface, TRAJECTORY_COLOR, start_pt, end_pt, 2)
        draw_arrow(surface, (min_x + 15, cy), start_pt) 
        draw_arrow(surface, (max_x - 15, cy), end_pt)

    elif moves_y:
        # === TRƯỜNG HỢP 2: DI CHUYỂN DỌC ===
        start_pt = (cx, min_y)
        end_pt = (cx, max_y)
        pygame.draw.line(surface, TRAJECTORY_COLOR, start_pt, end_pt, 2)
        draw_arrow(surface, (cx, min_y + 15), start_pt)
        draw_arrow(surface, (cx, max_y - 15), end_pt)

def draw_obstacle_shape(surface, obs_data):
    """Vẽ vật cản có hỗ trợ xoay, ép kiểu int để tránh lệch pixel"""
    # Ép kiểu int cho kích thước và tọa độ để tránh lỗi sub-pixel rendering
    w, h = int(obs_data['width']), int(obs_data['height'])
    x, y = int(obs_data['x']), int(obs_data['y'])
    angle = obs_data.get('angle', 0)
    is_static = obs_data.get('static', True)
    
    # Tạo surface con
    obs_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    
    color = BLACK if is_static else DYNAMIC_OBS_COLOR
    pygame.draw.rect(obs_surf, color, (0, 0, w, h))
    
    # Vẽ viền và tâm cho vật cản động
    if not is_static:
        pygame.draw.rect(obs_surf, BLACK, (0, 0, w, h), 2)
        pygame.draw.circle(obs_surf, WHITE, (w//2, h//2), 2)

    # Xoay surface
    if angle != 0:
        obs_surf = pygame.transform.rotate(obs_surf, angle)
    
    # Lấy rect mới canh giữa tại (x, y)
    new_rect = obs_surf.get_rect(center=(x, y))
    
    surface.blit(obs_surf, new_rect.topleft)

def draw_start_icon(surface, x, y, size=20):
    rect = pygame.Rect(x - size/2, y - size/2, size, size)
    pygame.draw.rect(surface, (200, 200, 200), (rect.x+2, rect.y+2, size, size), border_radius=4)
    pygame.draw.rect(surface, (50, 205, 50), rect, border_radius=4)
    pygame.draw.rect(surface, BLACK, rect, 2, border_radius=4)
    font = pygame.font.SysFont("arial", int(size), bold=True)
    text = font.render("S", True, WHITE)
    text_rect = text.get_rect(center=(x, y))
    surface.blit(text, text_rect)

def draw_goal_icon(surface, x, y, size=24):
    pygame.draw.circle(surface, BLACK, (x, y), size/2 + 1)
    pygame.draw.circle(surface, RED, (x, y), size/2)
    pygame.draw.circle(surface, WHITE, (x, y), size/3)
    pygame.draw.circle(surface, RED, (x, y), size/6)
    # Cờ
    pole_h = 30
    pygame.draw.line(surface, BLACK, (x + 8, y), (x + 8, y - pole_h), 2)
    pygame.draw.polygon(surface, RED, [(x+8, y-pole_h), (x+20, y-pole_h+6), (x+8, y-pole_h+12)])

# ======================================================================
# LOGIC CHÍNH
# ======================================================================
def select_map():
    print("\n--- MAP SELECTION ---")
    map_keys = list(maps.keys())
    for i, map_name in enumerate(map_keys):
        print(f"{i+1}. {map_name}")
    
    choice = input(f"Chọn bản đồ (1-{len(map_keys)}) hoặc Enter (Tất cả): ").strip()
    if choice == "": return map_keys
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(map_keys): return [map_keys[idx]]
    except: pass
    return [map_keys[0]]

def render_map_to_image(map_name):
    print(f"Đang vẽ: {map_name}...")
    map_data = maps[map_name]
    
    surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    surface.fill(WHITE)

    # 1. Vẽ Grid
    for x in range(ENV_PADDING, ENV_PADDING + ENV_SIZE + 1, CELL_SIZE):
        pygame.draw.line(surface, (230, 230, 230), (x, ENV_PADDING), (x, ENV_SIZE + ENV_PADDING))
    for y in range(ENV_PADDING, ENV_PADDING + ENV_SIZE + 1, CELL_SIZE):
        pygame.draw.line(surface, (230, 230, 230), (ENV_PADDING, y), (ENV_PADDING + ENV_SIZE, y))
    pygame.draw.rect(surface, BLACK, (ENV_PADDING, ENV_PADDING, ENV_SIZE, ENV_SIZE), 3)

    # 2. Phân loại vật cản
    static_obs = []
    dynamic_obs_data = []
    for obs_data in map_data["Obstacles"]:
        if obs_data.get('static', True):
            static_obs.append(obs_data)
        else:
            dynamic_obs_data.append(obs_data)

    # 3. Vẽ Trục Quỹ đạo (Nằm dưới cùng của layer vật cản)
    for obs_d in dynamic_obs_data:
        draw_trajectory_axis(surface, obs_d)

    # 4. Vẽ Vật cản Tĩnh (Có xử lý xoay)
    for obs_s in static_obs:
        draw_obstacle_shape(surface, obs_s)

    # 5. Vẽ Vật cản Động (Có xử lý xoay)
    for obs_d in dynamic_obs_data:
        draw_obstacle_shape(surface, obs_d)

    # 6. Vẽ Start / Goal (Grid Index -> Pixel)
    start_grid = map_data["Start"]
    goal_grid = map_data["Goal"]
    
    sx = ENV_PADDING + (start_grid[0] + 0.5) * CELL_SIZE
    sy = ENV_PADDING + (start_grid[1] + 0.5) * CELL_SIZE
    gx = ENV_PADDING + (goal_grid[0] + 0.5) * CELL_SIZE
    gy = ENV_PADDING + (goal_grid[1] + 0.5) * CELL_SIZE

    draw_start_icon(surface, sx, sy)
    draw_goal_icon(surface, gx, gy)

    # 7. Tên Map
    font = pygame.font.SysFont("arial", 24, bold=True)
    # Loại bỏ các chữ số ở cuối tên map (VD: uniform1 -> uniform)
    display_name = map_name.rstrip('0123456789')
    
    text = font.render(display_name, True, BLACK)
    
    # Đặt vị trí text lên vùng padding trên cùng (y = 5) để không đè vào map
    # ENV_PADDING thường là ~30px, đặt tại 5px là an toàn
    surface.blit(text, (ENV_PADDING, 5))
    
    # 8. Lưu ảnh
    output_dir = "map_images"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{map_name}.png")
    pygame.image.save(surface, filename)
    print(f" -> Đã lưu: {filename}")

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    
    selected_maps = select_map()
    for map_name in selected_maps:
        try:
            render_map_to_image(map_name)
        except Exception as e:
            print(f"Lỗi {map_name}: {e}")
            import traceback
            traceback.print_exc()
            
    pygame.quit()