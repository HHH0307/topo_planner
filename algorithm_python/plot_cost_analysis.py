import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.transforms import offset_copy
import numpy as np


def _setup_chinese_font() -> bool:
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]

    selected = None
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            selected = name
            break
        except Exception:
            continue

    if selected is not None:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False
    return selected is not None


def _build_data() -> Tuple[List[str], Dict[str, List[float]]]:
    maps = ["blocks", "default", "rooms", "corridor", "narrow_passage"]

    # 每个 map 对应 [Quadtree, Voronoi, Visibility]
    quadtree = [121.102, 111.673, 105.639, 155.344, 186.150]
    voronoi = [118.360, 115.499, 119.373, 163.579, 234.297]
    visibility = [102.958, 101.149, 93.682, 136.127, 167.832]

    costs = {
        "四叉树": quadtree,
        "维诺图": voronoi,
        "可视图": visibility,
    }
    return maps, costs


def _print_analysis(maps: List[str], costs: Dict[str, List[float]]) -> None:
    algo_names = list(costs.keys())
    matrix = np.array([costs[name] for name in algo_names])  # shape: [3, N]

    print("\n===== 路径 Cost 数据分析 =====")

    # 1) 每张地图最优算法
    print("\n[每张地图最优算法]")
    best_count = {name: 0 for name in algo_names}
    for i, map_name in enumerate(maps):
        values = matrix[:, i]
        best_idx = int(np.argmin(values))
        best_algo = algo_names[best_idx]
        best_value = values[best_idx]
        best_count[best_algo] += 1
        print(f"- {map_name}: {best_algo} (cost={best_value:.3f})")

    # 2) 各算法统计
    print("\n[各算法统计]")
    for algo in algo_names:
        values = np.array(costs[algo], dtype=float)
        print(
            f"- {algo}: 均值={values.mean():.3f}, 中位数={np.median(values):.3f}, "
            f"最小值={values.min():.3f}, 最大值={values.max():.3f}"
        )

    # 3) 胜出次数
    print("\n[胜出次数]")
    for algo in algo_names:
        print(f"- {algo}: {best_count[algo]} / {len(maps)}")

    # 4) 异常值提醒
    all_values = matrix.flatten()
    q1, q3 = np.percentile(all_values, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outliers = []
    for algo in algo_names:
        for map_name, v in zip(maps, costs[algo]):
            if v > upper_bound:
                outliers.append((map_name, algo, v))

    if outliers:
        print("\n[异常值提醒]")
        for map_name, algo, v in outliers:
            print(f"- {map_name} / {algo}: {v:.3f} (显著高于整体分布，建议复核原始数据)")


def _draw_bar_chart(maps: List[str], costs: Dict[str, List[float]], save_path: str, show: bool, use_chinese: bool) -> None:
    algo_names = list(costs.keys())

    if use_chinese:
        display_names = algo_names
        x_label = "地图"
        y_label = "Cost"
        y_label_log = "Cost（log尺度）"
        title = "不同地图下三种算法 Cost 对比"
    else:
        name_map = {
            "四叉树": "Quadtree",
            "维诺图": "Voronoi",
            "可视图": "Visibility",
        }
        display_names = [name_map.get(name, name) for name in algo_names]
        x_label = "Map"
        y_label = "Path Cost"
        y_label_log = "Path Cost (log scale)"
        title = "Path Cost Comparison Across Maps"
    x = np.arange(len(maps))
    width = 0.24

    values = np.array([costs[name] for name in algo_names])
    max_v = values.max()
    min_v = max(values[values > 0].min(), 1e-9)

    fig, ax = plt.subplots(figsize=(12, 6))

    offsets = [-width, 0.0, width]
    colors = ["#20417B", "#34D999", "#AA1D0A"]

    bar_containers = []
    for algo, label, offset, color in zip(algo_names, display_names, offsets, colors):
        bars = ax.bar(x + offset, costs[algo], width=width, label=label, color=color, alpha=0.9)
        bar_containers.append(bars)

    # 记录每组（每张地图）的最小 cost 位置，用于数值标注着色
    values_by_algo = [costs[name] for name in algo_names]
    min_positions = set()
    for map_idx in range(len(maps)):
        group_values = [values_by_algo[algo_idx][map_idx] for algo_idx in range(len(algo_names))]
        min_algo_idx = int(np.argmin(group_values))
        min_positions.add((min_algo_idx, map_idx))

    ax.set_xticks(x)
    ax.set_xticklabels(maps, rotation=10)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend()

    # 如果跨度过大，自动启用对数纵轴，避免小值柱子看不清
    if max_v / min_v > 50:
        ax.set_yscale("log")
        ax.set_ylabel(y_label_log)

    value_offset = offset_copy(ax.transData, fig=fig, x=0, y=4, units="points")
    for algo_idx, bars in enumerate(bar_containers):
        for map_idx, bar in enumerate(bars):
            height = bar.get_height()
            x_center = bar.get_x() + bar.get_width() / 2.0
            text_color = "#d62728" if (algo_idx, map_idx) in min_positions else "#222222"
            ax.text(
                x_center,
                height,
                f"{height:.3f}",
                transform=value_offset,
                ha="center",
                va="bottom",
                fontsize=8,
                color=text_color,
                rotation=0,
            )

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"\n图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and analyze topology algorithm path costs")
    parser.add_argument("--save", type=str, default="algorithm_python/cost_comparison_bar.png", help="保存图片路径")
    parser.add_argument("--no-show", action="store_true", help="不弹出图形窗口")
    args = parser.parse_args()

    use_chinese = _setup_chinese_font()
    if not use_chinese:
        print("[提示] 当前环境未检测到可用中文字体，图表将自动使用英文标签。")
    maps, costs = _build_data()
    _print_analysis(maps, costs)
    _draw_bar_chart(maps, costs, save_path=args.save, show=not args.no_show, use_chinese=use_chinese)


if __name__ == "__main__":
    main()
