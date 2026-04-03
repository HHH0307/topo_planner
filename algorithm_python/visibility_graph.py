import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from planner_common import (
	add_start_goal,
	astar_search,
	create_demo_map,
	draw_map,
	draw_path,
	get_available_maps,
	is_free_point,
	line_is_free,
	pairwise_indices,
	path_cost,
	setup_matplotlib_chinese_font,
)

setup_matplotlib_chinese_font()

Point = Tuple[float, float]

# 扩展障碍物角点的函数，在障碍物的四边扩展一个 margin 距离，返回扩展后的角点列表
def _expanded_corners(obstacle_rects, margin: float) -> List[Point]:
	points: List[Point] = []
	for x0, y0, x1, y1 in obstacle_rects:
		expanded = [
			(x0 - margin, y0 - margin),
			(x0 - margin, y1 + margin),
			(x1 + margin, y0 - margin),
			(x1 + margin, y1 + margin),
		]
		points.extend(expanded)
	return points

# 地图边界引导点的函数，在地图边界上均匀采样一些点作为可视图的候选节点
def _boundary_guides(width: int, height: int, pad: float = 3.0, step: float = 8.0) -> List[Point]:
	step = max(0.5, step)
	points: List[Point] = []
	x = pad
	while x <= width - pad + 1e-9:
		points.append((x, pad))
		points.append((x, height - pad))
		x += step

	y = pad + step
	while y < height - pad - 1e-9:
		points.append((pad, y))
		points.append((width - pad, y))
		y += step
	return points

# 自由空间引导点的函数，在地图边界附近均匀采样一些点作为可视图的候选节点
def _free_space_guides(map_data, samples_per_edge: int = 0, border_margin: float = 6.0) -> List[Point]:
	x_min = border_margin
	x_max = map_data.width - border_margin
	y_min = border_margin
	y_max = map_data.height - border_margin

	if samples_per_edge <= 0:
		n_h = max(2, min(5, int(round((x_max - x_min) / 35.0)) + 1))
		n_v = max(2, min(4, int(round((y_max - y_min) / 30.0)) + 1))
	else:
		n_h = max(1, int(samples_per_edge))
		n_v = n_h

	if n_h == 1:
		x_positions = [(x_min + x_max) / 2.0]
	else:
		x_positions = [x_min + (x_max - x_min) * i / (n_h - 1) for i in range(n_h)]

	if n_v == 1:
		y_positions = [(y_min + y_max) / 2.0]
	else:
		y_positions = [y_min + (y_max - y_min) * i / (n_v - 1) for i in range(n_v)]

	points: List[Point] = []
	# 上下边
	for x in x_positions:
		for y in (y_min, y_max):
			p = (x, y)
			if is_free_point(map_data, p, use_inflated=True):
				points.append(p)

	# 左右边（去掉角点，避免重复）
	if len(y_positions) > 2:
		for y in y_positions[1:-1]:
			for x in (x_min, x_max):
				p = (x, y)
				if is_free_point(map_data, p, use_inflated=True):
					points.append(p)
	return points

# 构建可视图图的函数，生成候选节点并连接可见的节点形成图结构
def build_visibility_graph(map_data=None, sample_step: float = 8.0, free_samples_per_edge: int = 0) -> Tuple[List[Point], Dict[int, List[Tuple[int, float]]], int, int, object]:
	map_data = map_data if map_data is not None else create_demo_map(map_name="default", inflation_radius=2)
	candidate_nodes = []

	for margin in (2.0, 3.0):
		for p in _expanded_corners(map_data.obstacle_rects, margin=margin):
			if is_free_point(map_data, p, use_inflated=True):
				candidate_nodes.append(p)

	for p in _boundary_guides(map_data.width, map_data.height, pad=3.0, step=sample_step):
		if is_free_point(map_data, p, use_inflated=True):
			candidate_nodes.append(p)

	for p in _free_space_guides(map_data, samples_per_edge=free_samples_per_edge, border_margin=6.0):
		candidate_nodes.append(p)

	nodes: List[Point] = []
	seen = set()
	for x, y in candidate_nodes:
		key = (round(x, 2), round(y, 2))
		if key in seen:
			continue
		seen.add(key)
		nodes.append((x, y))

	adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
	for i, j in pairwise_indices(len(nodes)):
		if line_is_free(map_data, nodes[i], nodes[j], step=0.4):
			w = math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
			adjacency[i].append((j, w))
			adjacency[j].append((i, w))

	start_idx = len(nodes)
	goal_idx = len(nodes) + 1
	nodes.extend([map_data.start, map_data.goal])

	def connect_special(idx: int, k: int = 18):
		p = nodes[idx]
		order = sorted(
			range(start_idx),
			key=lambda i: math.hypot(nodes[i][0] - p[0], nodes[i][1] - p[1]),
		)
		added = 0
		for i in order:
			if line_is_free(map_data, p, nodes[i], step=0.35):
				w = math.hypot(nodes[i][0] - p[0], nodes[i][1] - p[1])
				adjacency[idx].append((i, w))
				adjacency[i].append((idx, w))
				added += 1
			if added >= k:
				break

	connect_special(start_idx)
	connect_special(goal_idx)

	if line_is_free(map_data, map_data.start, map_data.goal, step=0.35):
		w = math.hypot(map_data.start[0] - map_data.goal[0], map_data.start[1] - map_data.goal[1])
		adjacency[start_idx].append((goal_idx, w))
		adjacency[goal_idx].append((start_idx, w))

	return nodes, adjacency, start_idx, goal_idx, map_data

# 求解可视图路径的函数，调用 A* 搜索算法找到从起点到终点的最短路径
def solve_visibility_graph(map_data=None, sample_step: float = 8.0, free_samples_per_edge: int = 0):
	nodes, adjacency, start_idx, goal_idx, map_data = build_visibility_graph(
		map_data=map_data,
		sample_step=sample_step,
		free_samples_per_edge=free_samples_per_edge,
	)
	idx_path = astar_search(nodes, adjacency, start_idx, goal_idx)
	path = [nodes[i] for i in idx_path] if idx_path else []
	return map_data, nodes, adjacency, path, path_cost(path)


# 绘制可视图的函数
def plot_visibility_graph(ax=None, show: bool = True, map_data=None, sample_step: float = 8.0, free_samples_per_edge: int = 0):
	map_data, nodes, adjacency, path, cost = solve_visibility_graph(
		map_data=map_data,
		sample_step=sample_step,
		free_samples_per_edge=free_samples_per_edge,
	)

	if ax is None:
		fig, ax = plt.subplots(figsize=(8, 5))
	else:
		fig = ax.figure

	draw_map(ax, map_data)
	for i, edges in adjacency.items():
		for j, _ in edges:
			if j <= i:
				continue
			ax.plot(
				[nodes[i][0], nodes[j][0]],
				[nodes[i][1], nodes[j][1]],
				color="#d9d9d9",
				linewidth=0.9,
				alpha=0.65,
			)

	if len(nodes) >= 2:
		graph_nodes = nodes[:-2]
	else:
		graph_nodes = nodes
	if graph_nodes:
		ax.scatter(
			[p[0] for p in graph_nodes],
			[p[1] for p in graph_nodes],
			s=10,
			color="#bfbfbf",
			alpha=0.75,
			zorder=4,
		)

	draw_path(ax, path)
	add_start_goal(ax, map_data.start, map_data.goal)
	ax.set_title(f"可视图 Visibility Graph\ncost: {cost:.3f}")

	if show:
		plt.tight_layout()
		plt.show()
	return fig, ax, cost


def _parse_point(text: str) -> Point:
	parts = text.split(",")
	if len(parts) != 2:
		raise argparse.ArgumentTypeError("Point format must be x,y")
	return float(parts[0]), float(parts[1])


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Visibility graph planner")
	parser.add_argument("--map", type=str, default="corridor", choices=get_available_maps(), help="Map preset name")
	parser.add_argument("--start", type=_parse_point, default=None, help="Start point, format x,y")
	parser.add_argument("--goal", type=_parse_point, default=None, help="Goal point, format x,y")
	parser.add_argument("--inflation", type=int, default=2, help="Obstacle inflation radius in cells")
	parser.add_argument("--sample-step", type=float, default=8.0, help="Boundary guide sampling step")
	parser.add_argument("--free-samples", type=int, default=0, help="Free-space guide samples per edge (0 for auto)")
	parser.add_argument("--save", type=str, default="", help="Save figure path (optional)")
	parser.add_argument("--no-show", action="store_true", help="Do not display GUI window")
	return parser


if __name__ == "__main__":
	args = _build_arg_parser().parse_args()
	setup_matplotlib_chinese_font()
	shared_map = create_demo_map(map_name=args.map, inflation_radius=args.inflation, start=args.start, goal=args.goal)
	fig, _, _ = plot_visibility_graph(
		show=not args.no_show,
		map_data=shared_map,
		sample_step=args.sample_step,
		free_samples_per_edge=args.free_samples,
	)
	if args.save:
		fig.savefig(args.save, dpi=150, bbox_inches="tight")
