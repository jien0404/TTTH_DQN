import subprocess
import sys

inputs = [
    ("test", 1 , 1,  6, 20),
    # ("test", 2 , 1,  1, 20),
    # ("test", 3 , 1,  1, 20),
    ("test", 4 , 1,  8, 20),
    ("test", 5 , 1,  2, 20),
]

controllers = {
        1: "DQNController",
        # 2: "DFQLController",
        # 3: "DWAController",
        4: "DQNPMController",
        5: "SDQNController",
    }

# Chạy các bài kiểm tra tự động
print("Bắt đầu chạy các bài kiểm tra tự động...")

for i, (mode, controller, map_id, models, num_test) in enumerate(inputs, 1):
    print(f"\n--- Chạy {controllers[controller]} trên map {map_id}  ---")
    input_str = f"{mode}\n{controller}\n{map_id}\n{models}\n{num_test}\n"
    result = subprocess.run(
        [sys.executable, 'main.py'],
        input=input_str,
        text=True,
        capture_output=True
    )
    print(result.stdout)
    if result.stderr:
        print("Lỗi:")
        print(result.stderr)

input_str = f"test\n6\n10\n20\n"
result = subprocess.run(
        [sys.executable, 'main.py'],
        input=input_str,
        text=True,
        capture_output=True
    )
print(result.stdout)
if result.stderr:
    print("Lỗi:")
    print(result.stderr)

print("Hoàn thành tất cả các bài kiểm tra.")
print("Bắt đầu thống kế kết quả")

maps = {
    1: "map1",
}

inputs_check = [
    (10, 'y')
]

for i, (map, output_files) in enumerate(inputs_check, 1):
    print(f'\n--- Thống kê {maps[i]} ---')
    input_str = f"{map}\n{output_files}\n"
    result = subprocess.run(
        [sys.executable, 'metrics.py'],
        input=input_str,
        text=True,
        capture_output=True
    )
    print(result.stdout)
    if result.stderr:
        print("Lỗi:")
        print(result.stderr)

print("Hoàn thành thống kê kết quả")
