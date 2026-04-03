import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

from planner_common import (
	MapData,
	add_start_goal,
	astar_search,
	create_demo_map,
	draw_map,
	draw_path,
	get_available_maps,
	is_free_point,
	line_is_free,
	path_cost,
	setup_matplotlib_chinese_font,
)

Point = Tuple[float, float]

# 收集地图边界上的样本点的函数，用于构建 Voronoi 图的候选节点，避免生成过多的节点导致计算量过大
def _collect_boundary_samples(binary_obstacle: np.ndarray, max_samples: int = 3000) -> np.ndarray:
	h, w = binary_obstacle.shape
	boundary_pts = []
	for y in range(1, h - 1):
		for x in range(1, w - 1):
			if not binary_obstacle[y, x]:
				continue
			local = binary_obstacle[y - 1 : y + 2, x - 1 : x + 2]
			if np.any(~local):
				boundary_pts.append((x, y))

	if len(boundary_pts) > max_samples:
		step = max(1, len(boundary_pts) // max_samples)
		boundary_pts = boundary_pts[::step]
	return np.array(boundary_pts, dtype=float)

# 判断点是否在自由空间的函数，检查点是否在地图范围内且在膨胀后的占用网格中为空闲
def _map_border_points(width: int, height: int, step: int = 5) -> np.ndarray:
	pts = []
	for x in range(0, width + 1, step):
		pts.append((x, 0))
		pts.append((x, height))
	for y in range(0, height + 1, step):
		pts.append((0, y))
		pts.append((width, y))
	return np.array(pts, dtype=float)

# 构建 Voronoi 图的函数，根据地图数据构建 Voronoi 图并连接可行的边形成图结构
def build_voronoi_graph(map_data=None) -> Tuple[List[Point], Dict[int, List[Tuple[int, float]]], int, int, MapData]:
	map_data = map_data if map_data is not None else create_demo_map(map_name="default", inflation_radius=2)
	samples = _collect_boundary_samples(map_data.inflated_occupancy)
	border = _map_border_points(map_data.width - 1, map_data.height - 1)
	points = np.vstack([samples, border])

	vor = Voronoi(points)
	valid_vid = {}
	nodes: List[Point] = []

	for vid, (x, y) in enumerate(vor.vertices):
		p = (float(x), float(y))
		if not is_free_point(map_data, p, use_inflated=True):
			continue
		if x < 0 or y < 0 or x >= map_data.width or y >= map_data.height:
			continue
		valid_vid[vid] = len(nodes)
		nodes.append(p)

	adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
	for ridge in vor.ridge_vertices:
		if len(ridge) != 2:
			continue
		a, b = ridge
		if a < 0 or b < 0 or a not in valid_vid or b not in valid_vid:
			continue
		ia, ib = valid_vid[a], valid_vid[b]
		pa, pb = nodes[ia], nodes[ib]
		if not line_is_free(map_data, pa, pb, step=0.5):
			continue
		w = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
		adjacency[ia].append((ib, w))
		adjacency[ib].append((ia, w))

	start_idx = len(nodes)
	goal_idx = len(nodes) + 1
	nodes.extend([map_data.start, map_data.goal])

	def connect_special(special_idx: int, k: int = 12):
		p = nodes[special_idx]
		order = sorted(range(start_idx), key=lambda i: math.hypot(nodes[i][0] - p[0], nodes[i][1] - p[1]))
		added = 0
		for i in order:
			if line_is_free(map_data, p, nodes[i], step=0.4):
				w = math.hypot(nodes[i][0] - p[0], nodes[i][1] - p[1])
				adjacency[special_idx].append((i, w))
				adjacency[i].append((special_idx, w))
				added += 1
			if added >= k:
				break

	connect_special(start_idx)
	connect_special(goal_idx)

	return nodes, adjacency, start_idx, goal_idx, map_data

# 求解 Voronoi 路径的函数，调用 A* 搜索算法找到从起点到终点的最短路径
def solve_voronoi(map_data=None):
	nodes, adjacency, start_idx, goal_idx, map_data = build_voronoi_graph(map_data=map_data)
	idx_path = astar_search(nodes, adjacency, start_idx, goal_idx)
	path = [nodes[i] for i in idx_path] if idx_path else []
	return map_data, nodes, adjacency, start_idx, goal_idx, path, path_cost(path)

# 绘制 Voronoi 图的函数，根据构建的 Voronoi 图和路径绘制地图、节点、边和路径，并设置标题显示路径代价
def plot_voronoi(ax=None, show: bool = True, map_data=None):
	map_data, nodes, adjacency, start_idx, goal_idx, path, cost = solve_voronoi(map_data=map_data)

	if ax is None:
		fig, ax = plt.subplots(figsize=(8, 5))
	else:
		fig = ax.figure

	draw_map(ax, map_data)

	for i, edges in adjacency.items():
		for j, _ in edges:
			if j <= i:
				continue
			if i in (start_idx, goal_idx) or j in (start_idx, goal_idx):
				continue
			ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color="#d9d9d9", linewidth=1.0)

	draw_path(ax, path)
	add_start_goal(ax, map_data.start, map_data.goal)
	ax.set_title(f"维诺图 Voronoi\ncost: {cost:.3f}")

	if show:
		plt.tight_layout()
		plt.show()
	return fig, ax, cost

# 解析点坐标的函数，将字符串格式的点坐标解析为数值元组
def _parse_point(text: str) -> Point:
	parts = text.split(",")
	if len(parts) != 2:
		raise argparse.ArgumentTypeError("Point format must be x,y")
	return float(parts[0]), float(parts[1])

# 构建命令行参数解析器的函数，定义地图、起点、终点、障碍物膨胀等参数，并返回解析器对象
def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Voronoi-based planner")
	parser.add_argument("--map", type=str, default="corridor", choices=get_available_maps(), help="Map preset name")
	parser.add_argument("--start", type=_parse_point, default=None, help="Start point, format x,y")
	parser.add_argument("--goal", type=_parse_point, default=None, help="Goal point, format x,y")
	parser.add_argument("--inflation", type=int, default=2, help="Obstacle inflation radius in cells")
	parser.add_argument("--save", type=str, default="", help="Save figure path (optional)")
	parser.add_argument("--no-show", action="store_true", help="Do not display GUI window")
	return parser


if __name__ == "__main__":
	args = _build_arg_parser().parse_args()
	setup_matplotlib_chinese_font()
	shared_map = create_demo_map(map_name=args.map, inflation_radius=args.inflation, start=args.start, goal=args.goal)
	fig, _, _ = plot_voronoi(show=not args.no_show, map_data=shared_map)
	if args.save:
		fig.savefig(args.save, dpi=150, bbox_inches="tight")