import math
import random
import numpy as np
import collections
from collections import deque
from controller.Controller import Controller


class DWAController(Controller):
    """
    DWA + BFS escape:
    - Bình thường: dùng DWA (mô phỏng horizon vài bước, chọn hướng tốt).
    - Nếu bị kẹt (distance-to-goal không giảm nhiều bước / lặp lại ô cũ):
        -> bật chế độ escape: tìm đường bằng BFS trong bán kính nhỏ rồi đi theo.
    """

    def __init__(self, goal, cell_size, env_padding,
                 is_training=False, model_path=None,
                 clearance_weight=0.6, heading_weight=1.0, velocity_weight=0.1,
                 horizon_steps=5,
                 stuck_threshold=8, bfs_radius=15):
        super().__init__(goal, cell_size, env_padding, is_training, model_path)

        self.base_dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]

        self.clearance_weight = clearance_weight
        self.heading_weight = heading_weight
        self.velocity_weight = velocity_weight
        self.horizon_steps = max(1, horizon_steps)

        # trạng thái để phát hiện kẹt
        self.last_dir = (1, 0)
        self.last_goal_dist = None
        self.stuck_steps = 0
        self.stuck_threshold = stuck_threshold
        self.recent_cells = collections.deque(maxlen=20)
        self.bfs_radius = bfs_radius

    # ------------ utils ------------
    def _to_grid_dir(self, d):
        dx = int(round(d[0]))
        dy = int(round(d[1]))
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        if dx == 0 and dy == 0:
            return (1, 0)
        return (dx, dy)

    def _will_collide_one_step(self, robot, gx, gy, obstacles):
        try:
            return robot.is_grid_cell_blocked(obstacles, gx, gy)
        except Exception:
            return True

    # ------------ DWA core ------------
    def _simulate_trajectory(self, robot, d, obstacles, goal_grid):
        cx, cy = robot.grid_x, robot.grid_y
        dx, dy = d
        steps_free = 0
        gx, gy = cx, cy

        for step in range(1, self.horizon_steps + 1):
            nx = cx + dx * step
            ny = cy + dy * step
            if self._will_collide_one_step(robot, nx, ny, obstacles):
                return True, (gx, gy), steps_free
            gx, gy = nx, ny
            steps_free += 1

        return False, (gx, gy), steps_free

    def _evaluate_velocity(self, robot, d, obstacles, goal_grid):
        collision, final_pos, steps_free = self._simulate_trajectory(
            robot, d, obstacles, goal_grid
        )
        if collision and steps_free == 0:
            return -9999.0

        fx, fy = final_pos
        gx, gy = goal_grid
        dist = abs(fx - gx) + abs(fy - gy)
        heading_score = -dist
        clearance_score = steps_free

        vx, vy = d
        lx, ly = self.last_dir
        dot = vx * lx + vy * ly
        velocity_score = dot

        score = (
            self.heading_weight * heading_score
            + self.clearance_weight * clearance_score
            + self.velocity_weight * velocity_score
        )
        return score

    # ------------ BFS escape ------------
    def _bfs_escape_step(self, robot, obstacles, goal_grid):
        """
        BFS trong bán kính bfs_radius quanh robot.
        - Nếu tìm được đường tới goal -> đi bước đầu tiên.
        - Nếu không, chọn node trong vùng BFS gần goal nhất -> đi theo hướng về node đó.
        """
        start = (robot.grid_x, robot.grid_y)
        gx, gy = goal_grid

        q = deque()
        q.append(start)
        visited = {start}
        parent = {}

        best_node = start
        best_dist = abs(start[0] - gx) + abs(start[1] - gy)

        while q:
            x, y = q.popleft()

            # giới hạn bán kính
            if abs(x - start[0]) + abs(y - start[1]) > self.bfs_radius:
                continue

            cur_dist = abs(x - gx) + abs(y - gy)
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_node = (x, y)

            if (x, y) == goal_grid:
                best_node = (x, y)
                break

            for d in self.base_dirs:
                nx, ny = x + d[0], y + d[1]
                if (nx, ny) in visited:
                    continue
                try:
                    blocked = robot.is_grid_cell_blocked(obstacles, nx, ny)
                except Exception:
                    blocked = True
                if blocked:
                    continue
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))

        # nếu best_node vẫn là start -> bất lực
        if best_node == start:
            return None

        # truy vết đường từ start -> best_node, lấy bước đầu tiên
        cur = best_node
        path = []
        while cur != start:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        first = path[0]
        dx = first[0] - start[0]
        dy = first[1] - start[1]
        return self._to_grid_dir((dx, dy))

    # ------------ stuck detection ------------
    def _update_stuck_state(self, robot, goal_grid):
        gx, gy = goal_grid
        cur_dist = abs(robot.grid_x - gx) + abs(robot.grid_y - gy)

        # lưu lịch sử ô
        cell = (robot.grid_x, robot.grid_y)
        self.recent_cells.append(cell)

        if self.last_goal_dist is None:
            self.last_goal_dist = cur_dist
            return

        # nếu không tiến gần hơn -> tăng stuck_steps
        if cur_dist >= self.last_goal_dist:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        self.last_goal_dist = cur_dist

        # nếu lặp lại cùng một ô nhiều lần -> cũng coi như bị kẹt
        if self.recent_cells.count(cell) >= 3:
            self.stuck_steps += 1

    def _is_stuck(self):
        return self.stuck_steps >= self.stuck_threshold

    # ------------ MAIN ------------
    def make_decision(self, robot, obstacles):
        gx_pixel, gy_pixel = self.goal

        goal_grid_x = int((gx_pixel - robot.env_padding) // robot.cell_size)
        goal_grid_y = int((gy_pixel - robot.env_padding) // robot.cell_size)
        goal_grid = (goal_grid_x, goal_grid_y)

        dist_goal_grid = abs(robot.grid_x - goal_grid_x) + abs(robot.grid_y - goal_grid_y)
        if dist_goal_grid == 0:
            return (0, 0)

        # cập nhật trạng thái kẹt
        self._update_stuck_state(robot, goal_grid)

        # 1) nếu đang kẹt -> thử BFS escape trước
        if self._is_stuck():
            escape_dir = self._bfs_escape_step(robot, obstacles, goal_grid)
            if escape_dir is not None:
                # reset một chút để tránh BFS spam
                self.stuck_steps = 0
                self.last_dir = escape_dir
                return escape_dir
            # nếu BFS cũng bó tay thì vẫn rơi xuống DWA ở dưới

        # 2) DWA bình thường
        best_dir = None
        best_score = -1e9

        for d in self.base_dirs:
            score = self._evaluate_velocity(robot, d, obstacles, goal_grid)
            if score > best_score:
                best_score = score
                best_dir = d

        if best_dir is None or best_score <= -9990:
            # cuối cùng vẫn tịt -> đứng yên
            return (0, 0)

        best_dir = self._to_grid_dir(best_dir)
        self.last_dir = best_dir
        return best_dir
