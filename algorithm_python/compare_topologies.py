import argparse
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from planner_common import create_demo_map, get_available_maps, setup_matplotlib_chinese_font
from quad_tree import plot_quadtree
from visibility_graph import plot_visibility_graph
from voronoi_diagram import plot_voronoi


def _parse_point(text: str):
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Point format must be x,y")
    return float(parts[0]), float(parts[1])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Voronoi / Visibility Graph / Quadtree planners")
    parser.add_argument("--map", type=str, default="pipeline", choices=get_available_maps(), help="Map preset name")
    parser.add_argument("--start", type=_parse_point, default=None, help="Start point, format x,y")
    parser.add_argument("--goal", type=_parse_point, default=None, help="Goal point, format x,y")
    parser.add_argument("--inflation", type=int, default=2, help="Obstacle inflation radius in cells")
    parser.add_argument("--save", type=str, default="", help="Save figure path (optional)")
    parser.add_argument("--no-show", action="store_true", help="Do not display GUI window")
    return parser


def _has_cjk_font() -> bool:
    candidates = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            return True
        except Exception:
            continue
    return False


def main():
    args = _build_arg_parser().parse_args()
    setup_matplotlib_chinese_font()
    shared_map = create_demo_map(map_name=args.map, inflation_radius=args.inflation, start=args.start, goal=args.goal)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    _, _, _ = plot_voronoi(ax=axes[0], show=False, map_data=shared_map)
    _, _, _ = plot_visibility_graph(ax=axes[1], show=False, map_data=shared_map)
    _, _, _ = plot_quadtree(ax=axes[2], show=False, map_data=shared_map)

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=160, bbox_inches="tight")

        save_root, save_ext = os.path.splitext(args.save)
        if not save_ext:
            save_ext = ".png"

        single_outputs = [
            ("voronoi", plot_voronoi),
            ("visibility", plot_visibility_graph),
            ("quadtree", plot_quadtree),
        ]

        for suffix, plot_fn in single_outputs:
            single_fig, _, _ = plot_fn(show=False, map_data=shared_map)
            single_path = f"{save_root}_{suffix}{save_ext}"
            single_fig.savefig(single_path, dpi=160, bbox_inches="tight")
            plt.close(single_fig)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
