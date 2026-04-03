import math
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from planner_common import (
	add_start_goal,
	astar_search,
	create_demo_map,
	draw_map,
	draw_path,
	get_available_maps,
	line_is_free,
	path_cost,
	setup_matplotlib_chinese_font,
)

Point = Tuple[float, float]

# 定义四叉树节点数据结构，包含矩形区域的坐标和子节点列表
@dataclass
class QuadNode:
	x0: float
	y0: float
	x1: float
	y1: float
	children: List["QuadNode"]
	state: int

	@property
	def center(self) -> Point:
		return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

# 计算四叉树节点的区域状态，1 表示全为障碍物，0 表示全为空闲，-1 表示混合状态
def _region_state(grid: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
	region = grid[y0:y1, x0:x1]
	if region.size == 0:
		return 0  # 空区域视为空闲
	if np.all(region):
		return 1
	if np.any(region):
		return -1
	return 0

# 构建四叉树的函数，递归地将地图划分为四个子区域，直到满足最小尺寸或区域状态一致
def _build_quadtree(grid: np.ndarray, x0: int, y0: int, x1: int, y1: int, min_size: int = 2) -> QuadNode:
	state = _region_state(grid, x0, y0, x1, y1)
	node = QuadNode(float(x0), float(y0), float(x1), float(y1), [], state)

	if state != -1:
		return node

	if (x1 - x0) <= min_size or (y1 - y0) <= min_size:
		node.state = 1 if np.mean(grid[y0:y1, x0:x1]) > 0.5 else 0
		return node

	mx = (x0 + x1) // 2
	my = (y0 + y1) // 2
	parts = [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]
	for px0, py0, px1, py1 in parts:
		child = _build_quadtree(grid, px0, py0, px1, py1, min_size=min_size)
		node.children.append(child)
	return node

# 递归收集四叉树的叶节点，将它们添加到输出列表中
def _collect_leaves(node: QuadNode, out: List[QuadNode]) -> None:
	if not node.children:
		out.append(node)
		return
	for c in node.children:
		_collect_leaves(c, out)

# 判断两个四叉树节点是否相邻的函数，判断它们是否在 x 或 y 方向上接触且在另一个方向上重叠
def _adjacent(a: QuadNode, b: QuadNode, eps: float = 1e-6) -> bool:
	x_touch = abs(a.x1 - b.x0) < eps or abs(b.x1 - a.x0) < eps
	y_overlap = min(a.y1, b.y1) - max(a.y0, b.y0) > eps
	y_touch = abs(a.y1 - b.y0) < eps or abs(b.y1 - a.y0) < eps
	x_overlap = min(a.x1, b.x1) - max(a.x0, b.x0) > eps
	return (x_touch and y_overlap) or (y_touch and x_overlap)

# 构建四叉树图的函数，根据地图数据构建四叉树并连接相邻的自由叶节点形成图结构
def build_quadtree_graph(map_data=None):
	map_data = map_data if map_data is not None else create_demo_map(map_name="default", inflation_radius=2)
	root = _build_quadtree(map_data.inflated_occupancy, 0, 0, map_data.width, map_data.height, min_size=2)
	leaves: List[QuadNode] = []
	_collect_leaves(root, leaves)
	free_leaves = [leaf for leaf in leaves if leaf.state == 0]

	nodes: List[Point] = [leaf.center for leaf in free_leaves]
	adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

	# 连接相邻的自由叶节点
	for i in range(len(free_leaves)):
		for j in range(i + 1, len(free_leaves)):
			if not _adjacent(free_leaves[i], free_leaves[j]):
				continue
			p, q = nodes[i], nodes[j]
			if not line_is_free(map_data, p, q, step=0.6):
				continue
			w = math.hypot(p[0] - q[0], p[1] - q[1])
			adjacency[i].append((j, w))
			adjacency[j].append((i, w))

	# 添加起点和终点
	start_idx = len(nodes)
	goal_idx = len(nodes) + 1
	nodes.extend([map_data.start, map_data.goal])

	# 连接起点和终点到附近的自由叶节点
	def connect_special(idx: int, radius: float = 20.0):
		p = nodes[idx]
		pairs = sorted(
			[
				(i, math.hypot(nodes[i][0] - p[0], nodes[i][1] - p[1]))
				for i in range(start_idx)
				if math.hypot(nodes[i][0] - p[0], nodes[i][1] - p[1]) <= radius
			],
			key=lambda x: x[1],
		)
		# 尝试连接更多节点，确保找到路径
		for i, w in pairs[:20]:
			if line_is_free(map_data, p, nodes[i], step=0.5):
				adjacency[idx].append((i, w))
				adjacency[i].append((idx, w))

	connect_special(start_idx)
	connect_special(goal_idx)
	return map_data, root, leaves, nodes, adjacency, start_idx, goal_idx

# 求解四叉树路径的函数，调用 A* 搜索算法找到从起点到终点的最短路径
def solve_quadtree(map_data=None):
	map_data, root, leaves, nodes, adjacency, start_idx, goal_idx = build_quadtree_graph(map_data=map_data)
	idx_path = astar_search(nodes, adjacency, start_idx, goal_idx)
	path = [nodes[i] for i in idx_path] if idx_path else []
	return map_data, root, leaves, nodes, adjacency, start_idx, goal_idx, path, path_cost(path)

# 绘制四叉树叶节点的函数，在图上绘制所有自由叶节点的边界框
def _draw_quadtree_leaves(ax, leaves: List[QuadNode]):
	for leaf in leaves:
		if leaf.state != 0:
			continue
		ax.add_patch(
			patches.Rectangle(
				(leaf.x0, leaf.y0),
				leaf.x1 - leaf.x0,
				leaf.y1 - leaf.y0,
				fill=False,
				edgecolor="#c77f45",
				linewidth=1.0,
				alpha=0.85,
			)
		)

# 绘制四叉树路径的函数，绘制四叉树图中的路径和起点终点，并设置标题显示路径代价
def plot_quadtree(ax=None, show: bool = True, map_data=None):
	map_data, root, leaves, nodes, adjacency, start_idx, goal_idx, path, cost = solve_quadtree(map_data=map_data)

	if ax is None:
		fig, ax = plt.subplots(figsize=(8, 5))
	else:
		fig = ax.figure

	draw_map(ax, map_data)
	_draw_quadtree_leaves(ax, leaves)

	for i, edges in adjacency.items():
		for j, _ in edges:
			if j <= i:
				continue
			ax.plot(
				[nodes[i][0], nodes[j][0]],
				[nodes[i][1], nodes[j][1]],
				color="#d3d3d3",
				linewidth=1.0,
				alpha=0.55,
			)

	if start_idx >= 0 and goal_idx >= 0:
		free_nodes = nodes[:start_idx]
		if free_nodes:
			ax.scatter(
				[p[0] for p in free_nodes],
				[p[1] for p in free_nodes],
				s=6,
				color="#c77f45",
				alpha=0.45,
				zorder=4,
			)

	draw_path(ax, path)
	add_start_goal(ax, map_data.start, map_data.goal)
	ax.set_title(f"四叉树 Quadtree\ncost: {cost:.3f}")

	if show:
		plt.tight_layout()
		plt.show()
	return fig, ax, cost

# 解析命令行参数的函数，定义地图、起点、终点、障碍物膨胀等参数，并返回解析器对象
def _parse_point(text: str) -> Point:
	parts = text.split(",")
	if len(parts) != 2:
		raise argparse.ArgumentTypeError("Point format must be x,y")
	return float(parts[0]), float(parts[1])

# 构建命令行参数解析器的函数，定义地图、起点、终点、障碍物膨胀等参数，并返回解析器对象
def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Quadtree planner")
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
	fig, _, _ = plot_quadtree(show=not args.no_show, map_data=shared_map)
	if args.save:
		fig.savefig(args.save, dpi=150, bbox_inches="tight")
  