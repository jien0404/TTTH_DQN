import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.plotting import table 

# Import dữ liệu map từ file của bạn
from MapData import maps

ENV_SIZE = 512
CELL_SIZE = 16
ENV_PADDING = int(ENV_SIZE * 0.06)

# ==============================================================================
# 1. CÁC HÀM TÍNH TOÁN METRIC
# ==============================================================================

def calculate_path_length(path):
    """Tính độ dài đường đi thực tế (Euclidean)."""
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

def calculate_path_avg_turn_deg(path):
    """
    Tính góc quay trung bình của đường đi (đơn vị Độ), logic giống Visualizer.
    """
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
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        
        # Dùng arctan2 để tính góc chính xác, sau đó lấy abs
        ang = abs(np.degrees(np.arctan2(cross, dot)))
        angles.append(ang)
        
    return np.mean(angles) if angles else 0.0

def read_paths_from_file(file_path):
    """Đọc file log, chuyển đổi toạ độ Grid -> Pixel để tính độ dài chính xác."""
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
                    grid_coords = eval(f"[{line}]")
                    
                    # QUY ĐỔI GRID -> PIXEL (Đồng bộ với Visualizer)
                    pixel_coords = [
                        (ENV_PADDING + (x + 0.5) * CELL_SIZE, 
                         ENV_PADDING + (y + 0.5) * CELL_SIZE) 
                        for x, y in grid_coords
                    ]
                    episodes.append(pixel_coords)
                except Exception as e:
                    pass
    return episodes, outcomes

def calculate_metrics_for_controller(episodes, outcomes):
    """Tính toán các thông số trung bình cho controller."""
    if not episodes:
        return {}

    num_episodes = len(episodes)
    success_count = outcomes.count("Success")
    
    successful_path_lengths = []
    successful_avg_angles = [] # Đổi tên biến cho rõ nghĩa

    for i, path in enumerate(episodes):
        if not path: continue
        
        # Chỉ tính metric cho các lần chạy thành công
        if outcomes[i] == "Success":
            path_len = calculate_path_length(path)
            
            # Sử dụng hàm tính góc mới (trả về độ trung bình)
            avg_turn_deg = calculate_path_avg_turn_deg(path)
            
            successful_path_lengths.append(path_len)
            successful_avg_angles.append(avg_turn_deg)
            
    # SỬA ĐOẠN CUỐI NÀY:
    success_rate = (success_count / num_episodes) * 100
    
    if successful_path_lengths:
        avg_success_path_length = np.mean(successful_path_lengths)
        avg_success_avg_angle = np.mean(successful_avg_angles)
    else:
        # Nếu không có lần nào thành công, gán NaN
        avg_success_path_length = np.nan
        avg_success_avg_angle = np.nan

    return {
        "Success Rate (%)": success_rate,
        "Avg Length (px)": avg_success_path_length,
        "Avg Angle (deg)": avg_success_avg_angle,
        "Runs": num_episodes
    }

# ==============================================================================
# 2. HÀM VẼ BÁO CÁO (Visualization)
# ==============================================================================

def create_scientific_table_image(df, title, output_path):
    """
    Vẽ bảng chuẩn LaTeX (Booktabs style) - Fixed MemoryError & FutureWarning
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    
    # 1. Tính toán kích thước Figure hợp lý
    num_rows = len(df)
    # Chiều cao: Header (2 đơn vị) + Data (num_rows) + Footer (1 đơn vị)
    # Mỗi đơn vị khoảng 0.5 inch
    fig_height = max(3, (num_rows + 4) * 0.5) 
    
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Quan trọng: Thiết lập giới hạn trục để dùng toạ độ data dễ dàng
    # Trục Y: chạy từ 0 (dưới cùng) đến n_rows + 3 (trên cùng)
    ylim_top = num_rows + 3
    ax.set_ylim(0, ylim_top)
    ax.set_xlim(0, 1) # Trục X chuẩn hoá 0-1
    ax.axis('off')    # Tắt khung trục mặc định
    
    # 2. Định nghĩa vị trí các cột (X coords)
    col_x = [0.05, 0.45, 0.65, 0.85]
    
    # 3. Định nghĩa vị trí các dòng (Y coords) - Vẽ từ trên xuống
    y_title = ylim_top - 0.5
    y_top_line = ylim_top - 1.0
    y_header_main = ylim_top - 1.4
    y_header_sub = ylim_top - 1.9
    y_mid_line = ylim_top - 2.2
    y_data_start = ylim_top - 2.8
    
    # ---------------------------------------------------------
    # VẼ HEADER
    # ---------------------------------------------------------
    
    # Tiêu đề bảng
    ax.text(0.5, y_title, title, ha='center', va='center', fontsize=16, weight='bold')
    
    # Kẻ ngang ĐẬM trên cùng (Dùng toạ độ data, KHÔNG dùng transform=ax.transAxes)
    ax.plot([0, 1], [y_top_line, y_top_line], color='black', linewidth=2)
    
    # Header Row 1: Controller & Metrics Group
    ax.text(col_x[0], y_header_main, "Controller", weight='bold', ha='left', va='center', fontsize=13)
    
    center_metric_x = (col_x[1] + col_x[-1]) / 2
    ax.text(center_metric_x, y_header_main, "Performance Metrics", weight='bold', ha='center', va='center', fontsize=13)
    # Gạch chân nhỏ dưới nhóm Metrics
    ax.plot([col_x[1]-0.05, col_x[-1]+0.05], [y_header_main-0.2, y_header_main-0.2], color='black', linewidth=0.8)

    # Header Row 2: Tên chi tiết
    sub_headers = ["", "Success (%)", "Length (px)", "Angle (deg)"]
    for i, txt in enumerate(sub_headers):
        if i == 0: continue
        ax.text(col_x[i], y_header_sub, txt, weight='bold', ha='center', va='center', fontsize=11)
        
    # Kẻ ngang MẢNH ngăn cách header và data
    ax.plot([0, 1], [y_mid_line, y_mid_line], color='black', linewidth=1)
    
    # ---------------------------------------------------------
    # VẼ DATA
    # ---------------------------------------------------------
    current_y = y_data_start
    
    # 1. Tìm Best Values (pandas min/max tự động bỏ qua NaN nên không cần sửa logic này)
    best_success = df.iloc[:, 0].max()
    best_len = df.iloc[:, 1].min()
    best_ang = df.iloc[:, 2].min()

    for controller_name, row in df.iterrows():
        val_success = row.iloc[0]
        val_len = row.iloc[1]
        val_ang = row.iloc[2]
        
        # Cột 1: Tên
        ax.text(col_x[0], current_y, str(controller_name), ha='left', va='center', fontsize=11)
        
        # Cột 2: Success
        w_suc = 'bold' if val_success == best_success else 'normal'
        ax.text(col_x[1], current_y, f"{val_success:.1f}", 
                ha='center', va='center', fontsize=11, weight=w_suc)
        
        # Cột 3: Length (Xử lý NaN)
        if pd.isna(val_len):
            txt_len = "N/A"
            w_len = 'normal'
        else:
            txt_len = f"{val_len:.2f}"
            w_len = 'bold' if val_len == best_len else 'normal'
            
        ax.text(col_x[2], current_y, txt_len, ha='center', va='center', fontsize=11, weight=w_len)
        
        # Cột 4: Angle (Xử lý NaN)
        if pd.isna(val_ang):
            txt_ang = "N/A"
            w_ang = 'normal'
        else:
            txt_ang = f"{val_ang:.2f}"
            w_ang = 'bold' if val_ang == best_ang else 'normal'
            
        ax.text(col_x[3], current_y, txt_ang, ha='center', va='center', fontsize=11, weight=w_ang)
        
        current_y -= 0.6
        
    # ---------------------------------------------------------
    # VẼ FOOTER
    # ---------------------------------------------------------
    # Kẻ ngang ĐẬM dưới cùng
    # Tính vị trí y dựa trên dòng cuối cùng
    y_bot_line = current_y + 0.3 
    ax.plot([0, 1], [y_bot_line, y_bot_line], color='black', linewidth=2)
    
    # Lưu ảnh
    # Bỏ tight_layout() vì chúng ta đã tự căn chỉnh toạ độ thủ công (manual layout)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved LaTeX-style Table to: {output_path}")

# def create_combined_chart(df, map_name, output_path):
#     """Vẽ 3 biểu đồ gộp chung vào 1 hình (Subplots sharing X axis)."""
    
#     metrics = ["Success Rate (%)", "Avg Length (px)", "Avg Angle (deg)"]
#     colors = ['#2ecc71', '#3498db', '#e67e22'] # Green, Blue, Orange
#     ylabels = ["Success Rate (%)", "Length (pixels)", "Turning (degrees)"]
#     titles = ["Success Rate (Higher is better)", "Path Length (Lower is better)", "Smoothness (Lower is better)"]

#     # Tạo 3 subplot xếp dọc, chia sẻ trục X
#     fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    
#     controllers = df.index
#     x = np.arange(len(controllers))

#     for i, metric in enumerate(metrics):
#         ax = axes[i]
#         values = df[metric]
        
#         # Vẽ bar chart
#         bars = ax.bar(x, values, color=colors[i], alpha=0.85, width=0.6, edgecolor='black')
        
#         # Grid và Label
#         ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
#         ax.set_ylabel(ylabels[i], fontsize=11, weight='bold')
#         ax.set_title(titles[i], fontsize=12, pad=5, color='#333333')
        
#         # Hiển thị giá trị trên đỉnh cột
#         y_max = values.max()
#         ax.set_ylim(0, y_max * 1.2) # Tăng giới hạn Y để có chỗ cho số
        
#         for bar in bars:
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height,
#                     f'{height:.1f}',
#                     ha='center', va='bottom', fontsize=10, weight='bold')

#     # Trục X chung ở biểu đồ cuối cùng
#     axes[-1].set_xlabel("Controller", fontsize=12, weight='bold')
#     axes[-1].set_xticks(x)
#     axes[-1].set_xticklabels(controllers, rotation=15, ha='right', fontsize=11)
    
#     # Tiêu đề chung cho cả ảnh
#     plt.suptitle(f"Performance Comparison on Map: {map_name}", fontsize=16, weight='bold', y=0.98)
    
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.93) # Chừa chỗ cho suptitle
    
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#     print(f"Saved Combined Chart to: {output_path}")

def create_combined_chart(df, map_name, output_path):
    """
    Vẽ Grouped Bar Chart (Cột ghép):
    - 3 metrics nằm cạnh nhau cho mỗi Controller.
    - Chuẩn hóa chiều cao (Normalized) để hiển thị cân đối.
    - Hiển thị giá trị thực (Real Value) trên đỉnh cột.
    """
    metrics = ["Success Rate (%)", "Avg Length (px)", "Avg Angle (deg)"]
    legend_labels = ["Success Rate (%)", "Avg Length (px)", "Avg Turn Angle (deg)"]
    colors = ['#2ecc71', '#3498db', '#e67e22'] # Xanh lá, Xanh dương, Cam
    
    # Setup Figure (Chỉ dùng 1 trục ax duy nhất)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Cấu hình vị trí cột
    controllers = df.index
    x = np.arange(len(controllers)) # Vị trí gốc trên trục X
    width = 0.25                    # Độ rộng mỗi cột (3 cột * 0.25 = 0.75 < 1, có khoảng hở)
    
    # --- VÒNG LẶP VẼ 3 LOẠI CỘT ---
    for i, metric in enumerate(metrics):
        # Lấy giá trị thực
        raw_values = df[metric]
        
        # CHUẨN HÓA: Chia cho giá trị lớn nhất để đưa về khoảng [0, 1]
        # Giúp các cột có chiều cao tương đồng nhau về mặt thị giác
        max_val = raw_values.max() if raw_values.max() > 0 else 1
        normalized_values = raw_values / max_val
        
        # Tính toán vị trí x lệch nhau: 
        # i=0 -> lệch trái (-0.25), i=1 -> giữa (0), i=2 -> lệch phải (+0.25)
        offset = (i - 1) * width 
        
        # Vẽ cột (Dùng chiều cao chuẩn hóa)
        bars = ax.bar(x + offset, normalized_values, width, 
                      label=legend_labels[i], color=colors[i], edgecolor='white', linewidth=1)
        
        # Ghi GIÁ TRỊ THỰC lên đầu cột
        for bar, val in zip(bars, raw_values):
            height = bar.get_height()
            # Xử lý format số liệu (Success/Angle lấy 1 số lẻ, Length lấy 0 số lẻ)
            val_str = f"{val:.1f}"
            
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    val_str,
                    ha='center', va='bottom', fontsize=9, weight='bold', color='#333333')

    # --- TRANG TRÍ ---
    ax.set_xlabel("Controller", fontsize=12, weight='bold')
    ax.set_title(f"Performance Comparison on Map: {map_name}\n(Visual height is normalized, numbers show real values)", 
                 fontsize=14, weight='bold', pad=15)
    
    # Set trục X
    ax.set_xticks(x)
    ax.set_xticklabels(controllers, rotation=15, ha='right', fontsize=11)
    
    # Ẩn trục Y số (vì chiều cao đã bị chuẩn hóa, số bên trái không còn ý nghĩa thực)
    ax.set_yticks([])
    ax.set_ylabel("Relative Performance Scale", fontsize=11, color='gray')
    
    # Tạo Legend phía dưới
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, frameon=False, fontsize=11)
    
    # Grid ngang mờ
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Lưu ảnh
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Grouped Bar Chart to: {output_path}")

# ==============================================================================
# 3. MAIN
# ==============================================================================

def select_map():
    print("Available maps:")
    map_keys = list(maps.keys())
    for i, map_name in enumerate(map_keys):
        print(f"{i+1}. {map_name}")
    while True:
        try:
            choice = int(input(f"Select map to analyze (1-{len(map_keys)}): "))
            if 1 <= choice <= len(map_keys):
                return map_keys[choice-1]
        except ValueError:
            pass

def main():
    selected_map = select_map()
    base_results_path = f"results_report_2/{selected_map}"
    
    if not os.path.exists(base_results_path):
        print(f"Results directory not found: {base_results_path}")
        return

    # 1. Thu thập dữ liệu
    all_metrics = []
    controller_folders = [d for d in os.listdir(base_results_path) if os.path.isdir(os.path.join(base_results_path, d))]
    
    print(f"\nProcessing {len(controller_folders)} controllers for map '{selected_map}'...")

    for controller_name in controller_folders:
        controller_path = os.path.join(base_results_path, controller_name)
        
        # Lấy file robot_paths mới nhất
        path_files = [f for f in os.listdir(controller_path) if f.startswith("robot_paths_") and f.endswith(".txt")]
        if not path_files: continue
        path_files.sort()
        latest_file = os.path.join(controller_path, path_files[-1])
        
        episodes, outcomes = read_paths_from_file(latest_file)
        if not episodes: continue

        met = calculate_metrics_for_controller(episodes, outcomes)
        met["Controller"] = controller_name
        all_metrics.append(met)

    if not all_metrics:
        print("No valid data found.")
        return

    # 2. Tạo DataFrame
    df = pd.DataFrame(all_metrics)
    df = df.set_index("Controller")
    # SẮP XẾP: Ưu tiên Success Rate (Giảm dần), sau đó đến Avg Length (Tăng dần)
    # Controller tốt nhất sẽ lên đầu
    df = df.sort_values(by=["Success Rate (%)", "Avg Length (px)"], ascending=[False, True])

    # Setup thư mục output
    output_dir = f"metrics_report_2/{selected_map}"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 3. Lưu CSV
    csv_path = os.path.join(output_dir, f"stats_{timestamp}.csv")
    df.to_csv(csv_path)
    print(f"\nSaved CSV to: {csv_path}")

    # 4. Vẽ Bảng (Scientific Table Image)
    table_img_path = os.path.join(output_dir, f"table_comparison_{timestamp}.png")
    create_scientific_table_image(df, f"Metrics Summary - {selected_map}", table_img_path)

    # 5. Vẽ Biểu đồ Gộp (Combined Chart)
    chart_path = os.path.join(output_dir, f"chart_comparison_{timestamp}.png")
    create_combined_chart(df, selected_map, chart_path)

    print("\nDone! Check the output directory.")

if __name__ == "__main__":
    main()