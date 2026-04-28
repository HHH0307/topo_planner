/**
 * @file visibility_graph_builder.h
 * @brief 可见性图策略：复用 FAR Planner 的 ContourDetector + ContourGraph
 *
 * 工作流与原版 FAR Planner 完全一致：
 *   1. ContourDetector 将障碍物点云投影到 2D 图像，用 OpenCV 提取多边形轮廓
 *   2. 取凸顶点作为候选 CTNode
 *   3. ContourGraph::MatchContourWithNavGraph 与全局图匹配
 *   4. ContourGraph::ExtractGlobalContours 刷新全局轮廓集合
 */

#pragma once

#include "far_planner/topo_graph_builder.h"
#include "far_planner/contour_detector.h"
#include "far_planner/contour_graph.h"
#include "far_planner/map_handler.h"

struct VisibilityBuilderParams {
    VisibilityBuilderParams() = default;
    ContourDetectParams cdetect_params;
    ContourGraphParams  cg_params;
};

class VisibilityGraphBuilder : public TopoGraphBuilder {
public:
    VisibilityGraphBuilder() = default;
    ~VisibilityGraphBuilder() override = default;

    void SetParams(const VisibilityBuilderParams& params) { params_ = params; }

    // ── TopoGraphBuilder 接口实现 ──────────────────────────────────────

    void Init(const rclcpp::Node::SharedPtr& nh) override {
        nh_ = nh;
        contour_detector_.Init(params_.cdetect_params);
        contour_graph_.Init(nh_, params_.cg_params);
        RCLCPP_INFO(nh_->get_logger(), "[TopoBuilder] VisibilityGraphBuilder initialized.");
    }

    void BuildTopoNodes(const NavNodePtr&        odom_node_ptr,
                        const PointCloudPtr&      obs_cloud,
                        CTNodeStack&              new_ctnodes,
                        std::vector<PointStack>&  realworld_contour) override
    {
        contour_detector_.BuildTerrainImgAndExtractContour(
            odom_node_ptr, obs_cloud, realworld_contour);
        contour_graph_.UpdateContourGraph(odom_node_ptr, realworld_contour);
        new_ctnodes = ContourGraph::contour_graph_;
    }

    void MatchWithNavGraph(const NodePtrStack& global_nodes,
                           const NodePtrStack& near_nodes,
                           CTNodeStack&        new_convex_vertices) override
    {
        contour_graph_.MatchContourWithNavGraph(
            global_nodes, near_nodes, new_convex_vertices);
    }

    void ExtractGlobalStructure() override {
        contour_graph_.ExtractGlobalContours();
    }

    void Reset() override {
        contour_graph_.ResetCurrentContour();
    }

    // ── 访问器 ────────────────────────────────────────────────────────

    const std::vector<PointPair>& GetGlobalContour()    const override { return ContourGraph::global_contour_; }
    const std::vector<PointPair>& GetUnmatchedContour() const override { return ContourGraph::unmatched_contour_; }
    const std::vector<PointPair>& GetBoundaryContour()  const override { return ContourGraph::boundary_contour_; }
    const std::vector<PointPair>& GetLocalBoundary()    const override { return ContourGraph::local_boundary_; }
    const CTNodeStack&            GetCurrentCTNodes()   const override { return ContourGraph::contour_graph_; }

    void AdjustCTNodeHeight(const CTNodeStack& ctnodes) override {
        map_handler_ref_->AdjustCTNodeHeight(ctnodes);
    }

    /** 注入 MapHandler 引用（由主节点在 Init 后调用） */
    void SetMapHandlerRef(MapHandler* map_handler) { map_handler_ref_ = map_handler; }

    /** 暴露 ContourDetector 用于 OpenCV 可视化（可选） */
    ContourDetector& GetContourDetector() { return contour_detector_; }

    /** 暴露 ContourGraph 用于碰撞检测（IsNavNodesConnectFreePolygon 等） */
    ContourGraph& GetContourGraph() { return contour_graph_; }

private:
    rclcpp::Node::SharedPtr nh_;
    VisibilityBuilderParams params_;
    ContourDetector contour_detector_;
    ContourGraph    contour_graph_;
    MapHandler*     map_handler_ref_ = nullptr;
};
