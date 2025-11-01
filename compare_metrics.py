import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from MapData import maps

# ==============================================================================
# CÁC HÀM TÍNH TOÁN METRIC (Tái sử dụng từ script trước)
# ==============================================================================

def calculateDistanceToGoal(start, goal):
    """Tính khoảng cách lý tưởng từ start đến goal (diagonal distance)."""
    dx = abs(start[0] - goal[0])
    dy = abs(start[1] - goal[1])
    return dx + dy # Sử dụng Manhattan distance cho môi trường lưới

def calculate_path_length(path):
    """Tính độ dài đường đi thực tế."""
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

def get_sum_turning_angle(path):
    """Tính tổng góc quay."""
    total_angle = 0
    for i in range(1, len(path) - 1):
        p1 = np.array(path[i-1])
        p2 = np.array(path[i])
        p3 = np.array(path[i+1])
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            continue
            
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
        total_angle += angle
    return total_angle

def read_paths_from_file(file_path):
    """Đọc các đường đi từ file, trả về danh sách các đường đi và kết quả (outcome)."""
    episodes = []
    outcomes = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Episode"):
                if "(Goal)" in line or "(Success)" in line:
                    outcomes.append("Success")
                elif "(Trapped)" in line:
                    outcomes.append("Trapped")
                else:
                    outcomes.append("Collision")
            elif line:
                try:
                    if line.endswith(','):
                        line = line[:-1]
                    coords_list = eval(f"[{line}]")
                    episodes.append(coords_list)
                except Exception as e:
                    print(f"Could not parse line: {line}\nError: {e}")
                    continue
    return episodes, outcomes

def calculate_metrics_for_controller(episodes, outcomes, start, goal):
    """Tính toán các thông số cho tất cả episode của một controller."""
    if not episodes:
        return {}

    num_episodes = len(episodes)
    success_count = outcomes.count("Success")
    
    path_lengths = []
    total_angles = []
    
    successful_path_lengths = []
    successful_total_angles = []

    for i, path in enumerate(episodes):
        if not path: continue
        
        path_len = calculate_path_length(path)
        total_angle = get_sum_turning_angle(path)
        
        path_lengths.append(path_len)
        total_angles.append(total_angle)

        if outcomes[i] == "Success":
            successful_path_lengths.append(path_len)
            successful_total_angles.append(total_angle)
            
    success_rate = (success_count / num_episodes) * 100
    
    # Tính toán trên các lần chạy thành công
    avg_success_path_length = np.mean(successful_path_lengths) if successful_path_lengths else 0
    avg_success_total_angle = np.mean(successful_total_angles) if successful_total_angles else 0

    return {
        "Success Rate (%)": success_rate,
        "Avg Path Length (Success)": avg_success_path_length,
        "Avg Total Angle (Success)": avg_success_total_angle,
        "Total Episodes": num_episodes
    }

# ==============================================================================
# HÀM MAIN ĐỂ SO SÁNH
# ==============================================================================

def select_map():
    """Chọn bản đồ từ người dùng."""
    print("Available maps:")
    map_keys = list(maps.keys())
    for i, map_name in enumerate(map_keys):
        print(f"{i+1}. {map_name}")
    
    while True:
        try:
            choice = int(input(f"Select map to analyze (1-{len(map_keys)}): "))
            if 1 <= choice <= len(map_keys):
                return map_keys[choice-1]
            print(f"Please enter a number between 1 and {len(map_keys)}")
        except ValueError:
            print("Please enter a valid number")

def main():
    selected_map = select_map()
    
    base_results_path = f"results/{selected_map}"
    if not os.path.exists(base_results_path):
        print(f"Results directory not found: {base_results_path}")
        return

    all_metrics = []
    
    # 1. Tự động tìm tất cả các controller đã được test
    controller_folders = [d for d in os.listdir(base_results_path) if os.path.isdir(os.path.join(base_results_path, d))]

    if not controller_folders:
        print(f"No controller results found in {base_results_path}")
        return

    print("\nFound controllers:", ", ".join(controller_folders))
    print("-" * 50)

    # 2. Lặp qua từng controller để tính toán metric
    for controller_name in controller_folders:
        controller_path = os.path.join(base_results_path, controller_name)
        
        # Tìm file kết quả mới nhất
        path_files = [f for f in os.listdir(controller_path) if f.startswith("robot_paths_") and f.endswith(".txt")]
        if not path_files:
            print(f"No path files for {controller_name}. Skipping.")
            continue
        
        path_files.sort()
        latest_file = path_files[-1]
        file_path = os.path.join(controller_path, latest_file)
        
        print(f"Processing {controller_name} using file: {latest_file}")
        
        start_pos = maps[selected_map]["Start"]
        goal_pos = maps[selected_map]["Goal"]
        
        episodes, outcomes = read_paths_from_file(file_path)
        metrics = calculate_metrics_for_controller(episodes, outcomes, start_pos, goal_pos)
        
        if metrics:
            metrics["Controller"] = controller_name
            all_metrics.append(metrics)

    if not all_metrics:
        print("\nNo valid data to generate report.")
        return

    df = pd.DataFrame(all_metrics)
    df = df.set_index("Controller") # Đặt tên controller làm chỉ số
    
    output_dir = f"metrics/{selected_map}"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sửa tên file thành .csv
    csv_path = os.path.join(output_dir, f"comparison_report_{timestamp}.csv")
    # Sửa hàm thành to_csv
    df.to_csv(csv_path)
    
    print(f"\nSaved comparison table to: {csv_path}")
    print("\n--- Comparison Table ---")
    print(df.round(2))
    print("------------------------")

    print("\nGenerating separate charts for each metric...")

    # Định nghĩa các metric cần vẽ và thông tin chi tiết
    metrics_to_plot = {
        "Success Rate (%)": {"note": "Higher is Better", "ylabel": "Success Rate (%)"},
        "Avg Path Length (Success)": {"note": "Lower is Better", "ylabel": "Average Path Length"},
        "Avg Total Angle (Success)": {"note": "Lower is Better", "ylabel": "Average Total Turning Angle (Radians)"}
    }

    colors = plt.cm.viridis(np.linspace(0, 1, len(df.index)))

    # Vòng lặp để tạo từng biểu đồ
    for metric_name, details in metrics_to_plot.items():
        # Tạo một figure mới cho mỗi biểu đồ
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Lấy dữ liệu và vẽ biểu đồ cột
        metric_values = df[metric_name]
        bars = ax.bar(df.index, metric_values, color=colors)
        
        # Thiết lập tiêu đề và các nhãn
        ax.set_title(f'Comparison: {metric_name} on Map "{selected_map}"', fontsize=15)
        ax.set_ylabel(details["ylabel"])
        ax.set_xlabel("Controller")
        
        # SỬA Ở ĐÂY: Tách việc xoay và căn lề ra
        # 1. Đặt các nhãn (ticks) trên trục x
        ax.set_xticks(np.arange(len(df.index)))
        # 2. Đặt nội dung cho các nhãn đó và TÙY CHỈNH chúng
        ax.set_xticklabels(df.index, rotation=30, ha="right")

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Thêm ghi chú "Higher/Lower is Better" vào góc
        ax.text(0.98, 0.98, details["note"], transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        # Thêm số liệu trên đầu mỗi cột
        for bar in bars:
            yval = bar.get_height()
            # Đặt giới hạn trục Y để có không gian cho text
            ax.set_ylim(top=ax.get_ylim()[1] * 1.1) 
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

        # Tạo tên file động và lưu
        safe_filename = metric_name.replace(" (%)", "").replace(" ", "_").lower()
        plot_path = os.path.join(output_dir, f"comparison_{safe_filename}_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Saved chart to: {plot_path}")
        plt.close(fig) # Đóng figure để giải phóng bộ nhớ trước khi tạo cái mới

if __name__ == "__main__":
    main()