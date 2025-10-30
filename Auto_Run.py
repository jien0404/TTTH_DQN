import subprocess
import sys

inputs = [
    ("train", 1, 11, 10000, "y"),
    ("train", 2, 11, 7000, "y"),
    ("train", 3, 11, 10000, "y"),
    ("train", 4, 11, 20000, "y"),
    ("train", 5, 11, 20000, "y"),
]

controllers = {
        1: "ClassicalQL",
        2: "DQNController",
        3: "DFQLController",
        4: "CombinedQLController",
        5: "DualQLController",
        6: "DWAController",
    }

maps = {
    1: "map1",
    2: "map2",
    3: "map3",
    4: "map4",
    5: "map5",
    6: "map6",
    7: "Sanghai0",
    8: "Denver",
    9: "Shanghai1",
    10: "Denver1", 
    11: "Denver2",
}

for i, (mode, controller, map_id, episodes, is_test) in enumerate(inputs, 1):
    print(f"\n--- Chạy {episodes} episodes với {controllers[controller]} trên {maps[map_id]}  ---")
    input_str = f"{mode}\n{controller}\n{map_id}\n{episodes}\n{is_test}\n"
    result = subprocess.run(
        [sys.executable, 'main.py'],
        input=input_str,
        text=True,
        capture_output=True
    )
    if result.stderr:
        print("Lỗi:")
        print(result.stderr)
