import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
import numpy as np

Point = Tuple[float, float]                 # (x, y) 坐标点
Rect = Tuple[float, float, float, float]    # (x0, y0, x1, y1) - 左下角和右上角坐标

# 定义地图数据结构，包含地图尺寸、障碍物列表、占用网格等信息
@dataclass
class MapData:
    width: int
    height: int
    obstacle_rects: List[Rect]
    occupancy: np.ndarray
    inflated_occupancy: np.ndarray
    start: Point
    goal: Point


# 定义地图边界墙的函数，生成四条边界墙的矩形表示
def _border_walls(width: int = 100, height: int = 60) -> List[Rect]:
    return [
        (0, 0, width, 2),
        (0, height - 2, width, height),
        (0, 0, 2, height),
        (width - 2, 0, width, height),
    ]


# 预设地图 定义，包含起点、终点和障碍物列表
MAP_PRESETS: Dict[str, Dict[str, object]] = {
    "default": {
        "start": (10.0, 10.0),
        "goal": (88.0, 45.0),
        "obstacles": _border_walls() + [
            (18, 28, 40, 32),
            (38, 14, 42, 32),
            (48, 14, 70, 18),
            (48, 18, 52, 36),
            (52, 42, 72, 46),
            (70, 18, 74, 46),
            (78, 0, 82, 32),
            (86, 8, 90, 32),
        ],
    },
    "rooms": {
        "start": (8.0, 8.0),
        "goal": (92.0, 52.0),
        "obstacles": _border_walls() + [
            (33, 2, 35, 16),
            (33, 30, 35, 44),
            (33, 52, 35, 58),
            (66, 2, 68, 12),
            (66, 26, 68, 40),
            (66, 50, 68, 58),
            (2, 20, 20, 22),
            (30, 20, 60, 22),
            (70, 20, 98, 22),
            (2, 40, 12, 42),
            (22, 40, 50, 42),
            (60, 40, 98, 42),
            (47, 27, 53, 33),
        ],
    },
    "corridor": {
        "start": (8.0, 8.0),
        "goal": (92.0, 52.0),
        "obstacles": _border_walls() + [
            (2, 28, 46, 32),
            (54, 28, 98, 32),
            (24, 2, 28, 14),
            (24, 22, 28, 58),
            (72, 2, 76, 38),
            (72, 46, 76, 58),
            (40, 8, 44, 24),
            (56, 36, 60, 52),
        ],
    },
    "pipeline": {
        "start": (8.0, 26.0),
        "goal": (92.0, 36.0),
        "obstacles": _border_walls() + [
            (2, 16, 32, 20),
            (2, 34, 32, 38),
            (32, 20, 64, 24),
            (32, 38, 64, 42),
            (64, 26, 98, 30),
            (64, 44, 98, 48),
        ],
    },
    "blocks": {
        "start": (8.0, 6.0),
        "goal": (92.0, 52.0),
        "obstacles": _border_walls() + [
            (18, 10, 30, 22),
            (36, 8, 48, 20),
            (54, 12, 66, 24),
            (72, 10, 84, 22),
            (20, 34, 32, 46),
            (40, 30, 52, 42),
            (60, 34, 72, 46),
            (78, 30, 90, 42),
            (44, 46, 56, 54),
        ],
    },
    "narrow_passage": {
        "start": (8.0, 8.0),
        "goal": (92.0, 52.0),
        "obstacles": _border_walls() + [
            (20, 2, 24, 38),
            (36, 22, 40, 58),
            (52, 2, 56, 38),
            (68, 22, 72, 58),
            (84, 2, 88, 38),
        ],
    },
    "deadend_maze": {
        "start": (8.0, 8.0),
        "goal": (92.0, 52.0),
        "obstacles": _border_walls() + [
            (2, 28, 46, 32),
            (54, 28, 98, 32),
            (24, 2, 28, 14),
            (24, 22, 28, 58),
            (72, 2, 76, 38),
            (72, 46, 76, 58),
            (12, 8, 20, 12),
            (8, 48, 16, 52),
            (42, 44, 50, 48),
            (58, 12, 66, 16),
            (84, 8, 92, 12),
            (80, 44, 88, 48),
        ],
    },
}


# 获取可用地图名称的函数，返回预设地图名称列表
def get_available_maps() -> List[str]:
    return list(MAP_PRESETS.keys())


# 膨胀占用网格的函数，将障碍物膨胀指定半径，生成新的占用网格
def _inflate_occupancy(occupancy: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return occupancy.copy()
    inflated = occupancy.copy()
    obstacle_y, obstacle_x = np.where(occupancy)
    for y, x in zip(obstacle_y, obstacle_x):
        y0 = max(0, y - radius)
        y1 = min(occupancy.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(occupancy.shape[1], x + radius + 1)
        inflated[y0:y1, x0:x1] = True
    return inflated


# 创建示例地图的函数，根据预设地图名称、膨胀半径、起点和终点生成地图数据
def create_demo_map(
    map_name: str = "default",
    inflation_radius: int = 2,
    start: Optional[Point] = None,
    goal: Optional[Point] = None,
) -> MapData:
    width, height = 100, 60
    if map_name not in MAP_PRESETS:
        raise ValueError(f"Unknown map '{map_name}'. Available maps: {', '.join(get_available_maps())}")

    preset = MAP_PRESETS[map_name]
    default_start = preset["start"]
    default_goal = preset["goal"]
    start = start if start is not None else default_start
    goal = goal if goal is not None else default_goal
    obstacle_rects: List[Rect] = list(preset["obstacles"])

    occupancy = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in obstacle_rects:
        occupancy[int(y0):int(y1), int(x0):int(x1)] = True

    inflated = _inflate_occupancy(occupancy, inflation_radius)
    map_data = MapData(width, height, obstacle_rects, occupancy, inflated, start, goal)
    if not is_free_point(map_data, start, use_inflated=True):
        raise ValueError(f"Start point {start} is invalid or collides with inflated obstacle")
    if not is_free_point(map_data, goal, use_inflated=True):
        raise ValueError(f"Goal point {goal} is invalid or collides with inflated obstacle")
    return map_data


# 解析点坐标的函数，将字符串格式的点坐标解析为数值元组
def is_free_point(map_data: MapData, p: Point, use_inflated: bool = True) -> bool:
    x, y = p
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= map_data.width or yi >= map_data.height:
        return False
    grid = map_data.inflated_occupancy if use_inflated else map_data.occupancy
    return not grid[yi, xi]


# 判断线段是否无碰撞的函数，在两点之间以一定步长采样，检查每个采样点是否在自由空间
def line_is_free(map_data: MapData, a: Point, b: Point, step: float = 0.5) -> bool:
    ax, ay = a
    bx, by = b
    dist = math.hypot(bx - ax, by - ay)
    num = max(2, int(dist / step))
    for i in range(num + 1):
        t = i / num
        x = ax + t * (bx - ax)
        y = ay + t * (by - ay)
        if not is_free_point(map_data, (x, y), use_inflated=True):
            return False
    return True

# 计算路径代价的函数，计算路径上相邻点之间的欧几里得距离之和
def path_cost(path: List[Point]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))

# A* 搜索算法的函数，使用启发式函数估计到目标的距离，优先探索代价较低的路径
def astar_search(nodes: List[Point], adjacency: Dict[int, List[Tuple[int, float]]], start_idx: int, goal_idx: int) -> List[int]:
    def heuristic(i: int) -> float:
        return math.hypot(nodes[i][0] - nodes[goal_idx][0], nodes[i][1] - nodes[goal_idx][1])

    pq: List[Tuple[float, int]] = [(heuristic(start_idx), start_idx)]
    g_cost = {start_idx: 0.0}
    parent: Dict[int, Optional[int]] = {start_idx: None}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal_idx:
            break
        for nxt, w in adjacency.get(cur, []):
            cand = g_cost[cur] + w
            if cand < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = cand
                parent[nxt] = cur
                heapq.heappush(pq, (cand + heuristic(nxt), nxt))

    if goal_idx not in parent:
        return []

    rev = [goal_idx]
    cur = goal_idx
    while parent[cur] is not None:
        cur = parent[cur]
        rev.append(cur)
    rev.reverse()
    return rev

# 以下是一些绘图和工具函数，用于在 Matplotlib 中绘制地图、路径和设置中文字体等
def draw_map(ax, map_data: MapData) -> None:
    inflated_img = np.where(map_data.inflated_occupancy, 1.0, np.nan)
    ax.imshow(
        inflated_img,
        origin="lower",
        extent=(0, map_data.width, 0, map_data.height),
        cmap="Purples",
        alpha=0.18,
        vmin=0,
        vmax=1,
    )

    for x0, y0, x1, y1 in map_data.obstacle_rects:
        ax.add_patch(
            patches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="black", edgecolor="black", linewidth=1.2)
        )

    ax.set_xlim(0, map_data.width)
    ax.set_ylim(0, map_data.height)
    ax.set_aspect("equal")
    ax.grid(False)


def draw_path(ax, path: List[Point], color: str = "#2eb82e") -> None:
    if len(path) < 2:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color=color, linewidth=2.0)


def add_start_goal(ax, start: Point, goal: Point) -> None:
    ax.scatter([start[0]], [start[1]], color="#00aaff", s=30, zorder=6)
    ax.scatter([goal[0]], [goal[1]], color="#ff7f0e", s=30, zorder=6)


def pairwise_indices(n: int) -> Iterable[Tuple[int, int]]:
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def setup_matplotlib_chinese_font() -> None:
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "Noto Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]

    available_fonts = sorted({f.name for f in font_manager.fontManager.ttflist})
    installed_preferred = [name for name in preferred_fonts if name in available_fonts]
    if not installed_preferred:
        cjk_keywords = ("cjk", "noto sans", "noto serif", "source han", "hei", "song", "kai", "fang")
        installed_preferred = [name for name in available_fonts if any(k in name.lower() for k in cjk_keywords)]
    if not installed_preferred:
        installed_preferred = ["DejaVu Sans"]

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = installed_preferred + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
