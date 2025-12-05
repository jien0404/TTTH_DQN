import pygame
import os
import numpy as np
from datetime import datetime
from MapData import maps
from Obstacle import Obstacle
from Colors import *

# ======================================================================
# THAM SỐ CHUNG
# ======================================================================
env_size = 512
cell_size = 16
env_padding = int(env_size * 0.06)

# Thêm không gian cho phần chú thích
LEGEND_AREA_HEIGHT = 160
WINDOW_WIDTH = env_size + 2 * env_padding
WINDOW_HEIGHT = env_size + 2 * env_padding + LEGEND_AREA_HEIGHT

PASTEL_COLORS = [
    (66, 135, 245),   # xanh dương nhạt
    (80, 200, 120),   # xanh lá pastel
    (255, 105, 97),   # đỏ nhạt
    (160, 108, 255),  # tím nhạt
    (255, 179, 186),  # hồng pastel
    (255, 127, 80),   # cam nhẹ
    (100, 149, 237),  # xanh cornflower
    (119, 221, 231),  # cyan pastel
    (186, 255, 201),  # xanh mint
    (255, 153, 204),  # hồng đậm pastel
]

# Màu dùng khi có chồng lấp (count >= 2)
OVERLAP_HIGHLIGHT = [
    (200, 30, 30),
    (255, 140, 0),
    (255, 215, 0),
    (34, 139, 34),
    (65, 105, 225),
    (138, 43, 226)
]

# ======================================================================
# HÀM TIỆN ÍCH
# ======================================================================
def select_map():
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
    length = 0.0
    for i in range(1, len(path)):
        p1 = np.array(path[i-1])
        p2 = np.array(path[i])
        length += np.linalg.norm(p2 - p1)
    return length

def path_turn_angles_deg(path):
    """Trả về danh sách các góc (deg, absolute) giữa các đoạn liên tiếp của path."""
    angles = []
    for i in range(1, len(path)-1):
        p0 = np.array(path[i-1])
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        dot = float(np.dot(v1, v2))
        # use stable atan2 for signed angle then abs
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        ang = abs(np.degrees(np.arctan2(cross, dot)))
        angles.append(ang)
    return angles

def read_all_paths_and_outcomes(file_path):
    """Đọc TẤT CẢ các đường đi và kết quả (outcome) từ file.""" 
    all_runs = []
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

def gather_controller_runs(controller_path):
    """Trả về list các runs của một controller (qua tất cả robot_paths_*.txt)."""
    all_runs = []
    path_files = [f for f in os.listdir(controller_path) if f.startswith("robot_paths_") and f.endswith(".txt")]
    for file_name in path_files:
        file_path = os.path.join(controller_path, file_name)
        runs = read_all_paths_and_outcomes(file_path)
        all_runs.extend(runs)
    return all_runs

def choose_best_effort_from_runs(runs):
    """Tương tự logic trước: nếu có Success -> chọn shortest success; else chọn longest failed."""
    successful = [r['path'] for r in runs if r['outcome'] == 'Success']
    failed = [r['path'] for r in runs if r['outcome'] != 'Success']
    if successful:
        return min(successful, key=calculate_path_length), "Success"
    if failed:
        return max(failed, key=calculate_path_length), "Failed"
    return None, None

def compute_controller_metrics(runs):
    """Trả về avg_length, avg_turn_angle, success_rate chỉ trên các run THÀNH CÔNG."""
    if not runs:
        return None, None, 0.0
        
    successful_runs = [r for r in runs if r.get('outcome') == 'Success']
    success_count = len(successful_runs)
    total_runs = len(runs)
    
    success_rate = float(success_count) / max(1, total_runs)
    
    if success_count == 0:
        # Không có lần nào thành công -> Length và Angle là None (N/A)
        return None, None, success_rate

    # Chỉ tính trung bình trên các lần thành công
    lengths = []
    mean_angles = []
    
    for r in successful_runs:
        p = r.get('path', [])
        # Length
        lengths.append(calculate_path_length(p))
        # Angle
        angles = path_turn_angles_deg(p)
        if angles:
            mean_angles.append(float(np.mean(angles)))
        else:
            mean_angles.append(0.0)
            
    avg_length = float(np.mean(lengths))
    avg_angle = float(np.mean(mean_angles))
    
    return avg_length, avg_angle, success_rate

# ======================================================================
# CLASS VISUALIZER
# ======================================================================
class PathVisualizer:
    def __init__(self, map_name, controllers_dict, base_color_map):
        """
        controllers_dict: {controller_name: {...}}
        base_color_map: {controller_name: (r,g,b)}
        """
        self.map_name = map_name
        self.controllers = controllers_dict
        self.base_color_map = base_color_map
        self.map_data = maps[map_name]
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f"Best Path Comparison on Map: {map_name}")
        self.font = pygame.font.SysFont("arial", 18, bold=True)
        self.small_font = pygame.font.SysFont("arial", 18)
        self.obstacles = [Obstacle(**obs_data) for obs_data in self.map_data["Obstacles"]]
        # occupancy map (pixel rounded -> set of controller names)
        self.point_to_controllers = {}
        self._build_occupancy_map()

    def _build_occupancy_map(self):
        self.point_to_controllers = {}
        for cname, data in self.controllers.items():
            path = data.get('path')
            if not path:
                continue
            visited = set()
            for (x, y) in path:
                key = (int(round(x)), int(round(y)))
                visited.add(key)
            for k in visited:
                if k not in self.point_to_controllers:
                    self.point_to_controllers[k] = set()
                self.point_to_controllers[k].add(cname)
        # compute counts (optional)
        self.point_counts = {k: len(v) for k, v in self.point_to_controllers.items()}
        self.max_overlap = max(self.point_counts.values()) if self.point_counts else 1

    def draw_environment_base(self):
        self.window.fill(WHITE)
        # draw grid
        for x in range(env_padding, env_padding + env_size + 1, cell_size):
            pygame.draw.line(self.window, (200,200,200), (x, env_padding), (x, env_size + env_padding))
        for y in range(env_padding, env_padding + env_size + 1, cell_size):
            pygame.draw.line(self.window, (200,200,200), (env_padding, y), (env_padding + env_size, y))

        for obs in self.obstacles:
            obs.draw(self.window)

        start_gx, start_gy = self.map_data["Start"]
        goal_gx, goal_gy = self.map_data["Goal"]
        start_px = (env_padding + (start_gx + 0.5) * cell_size, env_padding + (start_gy + 0.5) * cell_size)
        goal_px = (env_padding + (goal_gx + 0.5) * cell_size, env_padding + (goal_gy + 0.5) * cell_size)
        pygame.draw.rect(self.window, GREEN, (start_px[0]-6, start_px[1]-6, 12, 12))
        pygame.draw.circle(self.window, RED, (int(goal_px[0]), int(goal_px[1])), 9)
        pygame.draw.rect(self.window, BLACK, (env_padding, env_padding, env_size, env_size), 3)

    def _segment_color_by_overlap(self, controller_color, mid_point):
        key = (int(round(mid_point[0])), int(round(mid_point[1])))
        count = self.point_counts.get(key, 1)
        if count <= 1:
            return controller_color
        else:
            # map count 2 -> OVERLAP_HIGHLIGHT[0], 3-> [1], ...
            idx = min(count-2, len(OVERLAP_HIGHLIGHT)-1)
            return OVERLAP_HIGHLIGHT[idx]

    def draw_paths_segments(self):
        """Vẽ từng đoạn nhỏ; màu base lấy từ base_color_map.
           Khi một điểm đã bị vẽ nhiều lần, mỗi lần vẽ đè lên sẽ giảm width 30% (width *= 0.7^k).
        """
        # track how many times a given pixel key (rounded midpoint) đã bị vẽ
        drawn_counts = {}  # key -> times drawn so far
        base_width = 4

        # Important: iterate controllers in a stable order (insertion order of controllers dict)
        for cname in self.controllers.keys():
            data = self.controllers[cname]
            path = data.get('path')
            if not path or len(path) < 2:
                continue
            base_color = self.base_color_map.get(cname, (0, 0, 0))
            for i in range(1, len(path)):
                p1 = path[i-1]
                p2 = path[i]
                mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
                key = (int(round(mid[0])), int(round(mid[1])))

                # how many times this point was already drawn
                k = drawn_counts.get(key, 0)
                # shrink width by 30% per previous draw; ensure at least 1 px
                width = max(1, int(round(base_width * (0.5 ** k))))

                # choose color: if multiple controllers visit this point, use overlap highlight
                count = self.point_counts.get(key, 1)
                if count <= 1:
                    seg_color = base_color
                else:
                    idx = min(count-2, len(OVERLAP_HIGHLIGHT)-1)
                    seg_color = OVERLAP_HIGHLIGHT[idx]

                pygame.draw.line(self.window, seg_color, p1, p2, width)
                drawn_counts[key] = k + 1

    def draw_legend(self, mode='length'):
        """
        Vẽ legend 2 cột. Xử lý N/A và sắp xếp N/A xuống cuối.
        """
        entries = []
        for cname, data in self.controllers.items():
            if mode == 'length':
                val = data.get('avg_length') # Có thể là None
            elif mode == 'angle':
                val = data.get('avg_angle')  # Có thể là None
            elif mode == 'best':
                val = data.get('best_length')
                if val == float('inf'): val = None
            else:  # success
                val = data.get('success_rate', 0.0)
            entries.append((cname, data, val))

        # --- LOGIC SẮP XẾP ---
        # Helper để đẩy None xuống cuối khi sort tăng dần
        def sort_key_asc(item):
            val = item[2]
            return float('inf') if val is None else val

        # Helper để đẩy None xuống cuối khi sort giảm dần (Success)
        def sort_key_desc(item):
            val = item[2]
            return -1.0 if val is None else val

        if mode == 'success':
            # Success: Cao xếp trên
            entries.sort(key=sort_key_desc, reverse=True)
        else:
            # Length/Angle: Thấp xếp trên, None xếp cuối
            entries.sort(key=sort_key_asc)

        # --- CẤU HÌNH VẼ ---
        legend_x = env_padding
        legend_y = env_padding + env_size + 10
        line_h = 28
        col_icon_x = legend_x
        col_name_x = legend_x + 40
        col_val_x  = legend_x + 350

        # Header Title
        header_title = "Metric"
        if mode == 'length': header_title = "Avg Length (px)"
        elif mode == 'angle': header_title = "Avg Turn Angle"
        elif mode == 'best': header_title = "Best Length (px)"
        elif mode == 'success': header_title = "Success Rate"

        # Vẽ Header
        self.window.blit(self.font.render("Controller", True, BLACK), (col_name_x, legend_y))
        self.window.blit(self.font.render(header_title, True, BLACK), (col_val_x, legend_y))
        pygame.draw.line(self.window, BLACK, (legend_x, legend_y + 24), (WINDOW_WIDTH - env_padding, legend_y + 24), 2)
        
        current_y = legend_y + 35

        for cname, data, val in entries:
            base_color = self.base_color_map.get(cname, (0,0,0))
            
            # Xử lý text hiển thị
            if val is None:
                val_str = "N/A"
            else:
                if mode == 'length' or mode == 'best':
                    val_str = f"{val:.1f}"
                elif mode == 'angle':
                    val_str = f"{val:.1f}°"
                else:
                    val_str = f"{val:.1%}"

            # Vẽ dòng
            pygame.draw.line(self.window, base_color, (col_icon_x, current_y + 10), (col_icon_x + 28, current_y + 10), 8)
            self.window.blit(self.small_font.render(cname, True, BLACK), (col_name_x, current_y))
            self.window.blit(self.small_font.render(val_str, True, BLACK), (col_val_x, current_y))

            current_y += line_h

    def save_and_show(self, suffix_tag):
        output_dir = os.path.join("result_visualized_report", self.map_name)
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"best_paths_{self.map_name}_{suffix_tag}_{timestamp}.png")
        pygame.image.save(self.window, output_path)
        print(f"Saved visualization to: {output_path}")

    def render_and_save_versions(self):
        """Tạo 3 phiên bản: sort theo length, angle, success."""
        for mode in ['length', 'angle', 'success', 'best']:
            self.draw_environment_base()
            self.draw_paths_segments()
            self.draw_legend(mode=mode)
            self.save_and_show(suffix_tag=mode)
            # slight pause to ensure saved frame stable
            pygame.display.flip()

    def interactive_loop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.flip()
        pygame.quit()

# ======================================================================
# MAIN
# ======================================================================
def main():
    selected_map = select_map()
    base_results_path = f"results_report/{selected_map}"

    if not os.path.exists(base_results_path):
        print(f"Error: Results directory not found for map '{selected_map}'")
        return

    controller_folders = [d for d in os.listdir(base_results_path) if os.path.isdir(os.path.join(base_results_path, d))]
    controllers = {}

    print("\nGathering runs and computing metrics for each controller...")
    for controller_name in controller_folders:
        controller_path = os.path.join(base_results_path, controller_name)
        runs = gather_controller_runs(controller_path)
        if not runs:
            print(f"No runs for {controller_name}, skipping.")
            continue
        best_path, status = choose_best_effort_from_runs(runs)
        avg_len, avg_ang, succ_rate = compute_controller_metrics(runs)
        best_len = calculate_path_length(best_path) if best_path else float('inf')
        if status != "Success":
            best_len = float('inf')

        controllers[controller_name] = {
            'path': best_path,
            'status': status,
            'best_length': best_len,
            'runs': runs,
            'avg_length': avg_len,
            'avg_angle': avg_ang,
            'success_rate': succ_rate
        }
        # --- AFTER controllers dict is populated ---
        # Create a fixed color mapping so legend order won't break color association
        controller_names_order = list(controllers.keys())  # preserves insertion order
        base_color_map = {}
        for i, cname in enumerate(controller_names_order):
            base_color_map[cname] = PASTEL_COLORS[i % len(PASTEL_COLORS)]
        # --- SỬA Ở ĐÂY: Xử lý hiển thị khi giá trị là None (N/A) ---
        display_len = f"{avg_len:.1f}" if avg_len is not None else "N/A"
        display_ang = f"{avg_ang:.1f}" if avg_ang is not None else "N/A"
        print(f"{controller_name}: runs={len(runs)}, avg_len={display_len}, avg_ang={display_ang}, succ_rate={succ_rate:.2%}")

    if not controllers:
        print("No controllers with data found. Exiting.")
        return

    visualizer = PathVisualizer(selected_map, controllers, base_color_map)
    visualizer.render_and_save_versions()
    print("Visualization complete. Opening interactive window (close window to exit).")
    visualizer.interactive_loop()

if __name__ == "__main__":
    main()
