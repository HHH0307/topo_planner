/**
 * @file topo_graph_builder.h
 * @brief 通用拓扑图构建策略抽象接口
 *
 * 所有拓扑图构建策略（可见性图、Voronoi图等）均继承此接口。
 * 接口职责：
 *   给定当前障碍物点云和里程计节点，输出一批「候选轮廓节点」CTNode，
 *   由 DynamicGraph 统一进行节点生命周期管理。
 *
 * 不变的部分（沿用 FAR Planner）：
 *   - DynamicGraph：节点投票/合并/清除生命周期
 *   - GraphPlanner：A* 路径搜索
 *   - MapHandler / ScanHandler / TerrainPlanner：地图与传感器
 *   - GraphMsger / DPVisualizer：通信与可视化
 *   - AutoExplore：自主探索目标选取
 */

#pragma once

#include "far_planner/utility.h"
#include "far_planner/node_struct.h"

// ──────────────────────────────────────────────────────────────────────
// 策略枚举
// ──────────────────────────────────────────────────────────────────────
enum class TopoBuilderType {
    VISIBILITY_GRAPH = 0,  ///< FAR Planner 原始可见性图（凸多边形顶点）
    VORONOI_GRAPH    = 1,  ///< 基于障碍物点云的 Voronoi 骨架图
};

// ──────────────────────────────────────────────────────────────────────
// 抽象接口
// ──────────────────────────────────────────────────────────────────────
class TopoGraphBuilder {
public:
    TopoGraphBuilder() = default;
    virtual ~TopoGraphBuilder() = default;

    /**
     * @brief 初始化构建器（由子类实现）
     * @param nh ROS2 节点句柄
     */
    virtual void Init(const rclcpp::Node::SharedPtr& nh) = 0;

    /**
     * @brief 核心接口：根据当前障碍物点云和里程计，生成候选 CTNode
     *
     * @param odom_node_ptr    当前里程计导航节点
     * @param obs_cloud        当前障碍物点云（已经过 MapHandler 处理）
     * @param new_ctnodes[out] 本帧新增/更新的候选轮廓节点
     * @param realworld_contour[out] 当前帧真实轮廓（用于可视化 / ContourGraph 匹配）
     */
    virtual void BuildTopoNodes(
        const NavNodePtr&              odom_node_ptr,
        const PointCloudPtr&           obs_cloud,
        CTNodeStack&                   new_ctnodes,
        std::vector<PointStack>&       realworld_contour) = 0;

    /**
     * @brief 将当前帧的轮廓/骨架节点与全局导航图进行匹配
     *
     * @param global_nodes 当前全局导航节点（来自 DynamicGraph）
     * @param near_nodes   近邻节点（来自 DynamicGraph::GetExtendLocalNode）
     * @param new_convex_vertices[out] 真正的「新」节点候选
     */
    virtual void MatchWithNavGraph(
        const NodePtrStack& global_nodes,
        const NodePtrStack& near_nodes,
        CTNodeStack&        new_convex_vertices) = 0;

    /**
     * @brief 提取/刷新全局轮廓（边界检测，可视化用）
     *        可见性图：等同于 ContourGraph::ExtractGlobalContours()
     *        Voronoi 图：提取骨架边
     */
    virtual void ExtractGlobalStructure() = 0;

    /**
     * @brief 重置当前帧状态（每帧前调用）
     */
    virtual void Reset() = 0;

    // ── 访问器：不同策略暴露相同的只读引用供可视化和碰撞检测 ──

    /**
     * @brief 返回全局轮廓边（用于 planner_viz_ 和碰撞判断）
     */
    virtual const std::vector<PointPair>& GetGlobalContour()    const = 0;
    virtual const std::vector<PointPair>& GetUnmatchedContour() const = 0;
    virtual const std::vector<PointPair>& GetBoundaryContour()  const = 0;
    virtual const std::vector<PointPair>& GetLocalBoundary()    const = 0;

    /**
     * @brief 返回当前帧 CTNode 图（用于可视化）
     */
    virtual const CTNodeStack& GetCurrentCTNodes() const = 0;

    /**
     * @brief 调整 CTNode 高度（MapHandler 回调）
     */
    virtual void AdjustCTNodeHeight(const CTNodeStack& ctnodes) = 0;
};
