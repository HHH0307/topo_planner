/**
 * @file voronoi_graph_builder.h
 * @brief Voronoi 图策略：从障碍物点云近似计算 Voronoi 骨架节点
 *
 * 算法思路（近似 Voronoi 骨架，无需精确 Voronoi 计算）：
 *  1. 在以机器人为中心的 2D 栅格上，计算每个自由格到最近障碍物的距离
 *     （Distance Transform / KD-tree 查询）
 *  2. 取局部距离极大值点作为 Voronoi 骨架候选点
 *     （即在所有邻居中距离最大的格子 —— 等价于 Voronoi 脊线）
 *  3. 过滤：
 *     a. 距离阈值：点到障碍物距离 > robot_dim/2 才可通行
 *     b. 相邻候选点去重（间距 < voxel_dim * cluster_radius）
 *  4. 将候选点包装为 CTNode（free_direct = PILLAR，不携带轮廓方向）
 *     DynamicGraph 的投票/合并机制同样适用
 *
 * 与可见性图的对比：
 *  - 可见性图：节点在障碍物凸顶点处，路径紧贴障碍物
 *  - Voronoi 图：节点在自由空间最宽处，路径远离障碍物，更安全
 */

#pragma once

#include "far_planner/topo_graph_builder.h"
#include "far_planner/map_handler.h"

struct VoronoiBuilderParams {
    VoronoiBuilderParams() = default;
    float sensor_range     = 15.0f; ///< 分析半径 (m)
    float voxel_dim        = 0.3f;  ///< 骨架搜索网格分辨率 (m)
    float robot_clearance  = 0.5f;  ///< 骨架点到障碍物最小间距 (m) = robot_dim/2
    float cluster_radius   = 1.2f;  ///< 候选点聚类半径 (m)，避免骨架点过密
    float min_dist_to_obs  = 0.4f;  ///< 候选点到障碍物最小距离 (m)
    int   neighbor_window  = 2;     ///< 局部最大值检测窗口（格子数）
};

class VoronoiGraphBuilder : public TopoGraphBuilder {
public:
    VoronoiGraphBuilder() = default;
    ~VoronoiGraphBuilder() override = default;

    void SetParams(const VoronoiBuilderParams& params) { params_ = params; }

    // ──────────────────────────────────────────────────────────────────
    // TopoGraphBuilder 接口实现
    // ──────────────────────────────────────────────────────────────────

    void Init(const rclcpp::Node::SharedPtr& nh) override {
        nh_ = nh;
        RCLCPP_INFO(nh_->get_logger(), "[TopoBuilder] VoronoiGraphBuilder initialized.");
    }

    void BuildTopoNodes(const NavNodePtr&        odom_node_ptr,
                        const PointCloudPtr&      obs_cloud,
                        CTNodeStack&              new_ctnodes,
                        std::vector<PointStack>&  realworld_contour) override
    {
        new_ctnodes.clear();
        realworld_contour.clear();
        current_ctnodes_.clear();
        global_contour_.clear();
        unmatched_contour_.clear();

        if (odom_node_ptr == NULL || obs_cloud == nullptr || obs_cloud->empty()) return;

        odom_pos_ = odom_node_ptr->position;

        // 1. 建立障碍物 KD-tree（在 sensor_range 内）
        PointCloudPtr local_obs(new PointCloud());
        FilterLocalObsCloud(obs_cloud, local_obs);
        if (local_obs->empty()) return;

        obs_kdtree_.setInputCloud(local_obs);

        // 2. 在 2D 栅格上计算距离变换并寻找局部最大值
        std::vector<Point3D> skeleton_candidates;
        ExtractVoronoiSkeleton(local_obs, skeleton_candidates);
        if (skeleton_candidates.empty()) return;

        // 3. 聚类去重
        std::vector<Point3D> skeleton_filtered;
        ClusterFilter(skeleton_candidates, skeleton_filtered);

        // 4. 包装为 CTNode（Voronoi 节点没有轮廓方向，设为 PILLAR）
        for (const auto& p : skeleton_filtered) {
            CTNodePtr ctnode = std::make_shared<CTNode>();
            ctnode->position          = p;
            ctnode->free_direct       = NodeFreeDirect::PILLAR;
            ctnode->is_global_match   = false;
            ctnode->is_contour_necessary = true;
            ctnode->is_ground_associate  = true;
            // surf_dirs 未定义：PILLAR 类型不使用
            ctnode->surf_dirs = {Point3D(0,0,0), Point3D(0,0,0)};
            current_ctnodes_.push_back(ctnode);
        }
        new_ctnodes = current_ctnodes_;

        // 5. 将骨架边作为「轮廓」供可视化（连接相邻骨架点）
        BuildSkeletonEdges(skeleton_filtered, realworld_contour);
    }

    void MatchWithNavGraph(const NodePtrStack& global_nodes,
                           const NodePtrStack& near_nodes,
                           CTNodeStack&        new_convex_vertices) override
    {
        // Voronoi 节点匹配：与可见性图不同，Voronoi 骨架点无需检查视线遮挡，
        // 只需检查：1) 未被已有导航节点覆盖；2) 不在障碍物内部
        new_convex_vertices.clear();

        // 建立近邻节点的 KD-tree 以快速判断是否已有匹配
        PointCloudPtr nav_cloud(new PointCloud());
        for (const auto& np : near_nodes) {
            if (np == NULL) continue;
            PCLPoint p;
            p.x = np->position.x; p.y = np->position.y; p.z = np->position.z;
            nav_cloud->push_back(p);
        }

        bool has_near_graph = !nav_cloud->empty();
        pcl::KdTreeFLANN<PCLPoint> nav_kdtree;
        if (has_near_graph) nav_kdtree.setInputCloud(nav_cloud);

        for (auto& ctnode : current_ctnodes_) {
            if (ctnode == NULL) continue;

            // 已匹配则跳过
            if (ctnode->is_global_match) continue;

            // 检查是否与已有导航节点足够近（已被覆盖）
            if (has_near_graph) {
                PCLPoint pcl_p;
                pcl_p.x = ctnode->position.x;
                pcl_p.y = ctnode->position.y;
                pcl_p.z = ctnode->position.z;
                std::vector<int> pIdxK(1);
                std::vector<float> pdDistK(1);
                if (nav_kdtree.nearestKSearch(pcl_p, 1, pIdxK, pdDistK) > 0) {
                    if (pdDistK[0] < params_.cluster_radius * params_.cluster_radius) {
                        ctnode->is_global_match = true; // 标记为已覆盖，不产生新节点
                        continue;
                    }
                }
            }
            ctnode->is_contour_necessary = true;
            new_convex_vertices.push_back(ctnode);
        }
    }

    void ExtractGlobalStructure() override {
        // Voronoi：将骨架边存入 global_contour_（可视化用）
        // 已在 BuildTopoNodes 中通过 realworld_contour 填充 global_contour_
        // 此处将 current skeleton edges 转为 PointPair 格式
        // (已经在 BuildSkeletonEdges 中填充 global_contour_)
    }

    void Reset() override {
        current_ctnodes_.clear();
        global_contour_.clear();
        unmatched_contour_.clear();
        skeleton_edges_.clear();
    }

    // ── 访问器 ────────────────────────────────────────────────────────

    const std::vector<PointPair>& GetGlobalContour()    const override { return global_contour_; }
    const std::vector<PointPair>& GetUnmatchedContour() const override { return unmatched_contour_; }
    const std::vector<PointPair>& GetBoundaryContour()  const override { return boundary_contour_; }
    const std::vector<PointPair>& GetLocalBoundary()    const override { return local_boundary_; }
    const CTNodeStack&            GetCurrentCTNodes()   const override { return current_ctnodes_; }

    void AdjustCTNodeHeight(const CTNodeStack& ctnodes) override {
        if (map_handler_ref_) map_handler_ref_->AdjustCTNodeHeight(ctnodes);
    }

    void SetMapHandlerRef(MapHandler* map_handler) { map_handler_ref_ = map_handler; }

private:
    rclcpp::Node::SharedPtr nh_;
    VoronoiBuilderParams    params_;
    MapHandler*             map_handler_ref_ = nullptr;

    Point3D odom_pos_;
    CTNodeStack current_ctnodes_;

    std::vector<PointPair> global_contour_;
    std::vector<PointPair> unmatched_contour_;
    std::vector<PointPair> boundary_contour_;   // 保持空，Voronoi 不处理边界
    std::vector<PointPair> local_boundary_;     // 保持空

    std::vector<std::pair<Point3D, Point3D>> skeleton_edges_;

    pcl::KdTreeFLANN<PCLPoint> obs_kdtree_;

    // ── 私有算法函数 ──────────────────────────────────────────────────

    /** 过滤出传感器范围内的局部障碍物点云 */
    void FilterLocalObsCloud(const PointCloudPtr& obs_in, PointCloudPtr& obs_out) const {
        obs_out->clear();
        const float range_sq = params_.sensor_range * params_.sensor_range;
        for (const auto& pt : obs_in->points) {
            const float dx = pt.x - odom_pos_.x;
            const float dy = pt.y - odom_pos_.y;
            if (dx*dx + dy*dy < range_sq) {
                obs_out->push_back(pt);
            }
        }
    }

    /**
     * @brief 在 2D 网格上计算每个格子到最近障碍物的距离，
     *        找出局部最大值作为 Voronoi 骨架候选点
     */
    void ExtractVoronoiSkeleton(const PointCloudPtr&  local_obs,
                                std::vector<Point3D>& candidates) const
    {
        candidates.clear();
        const float res  = params_.voxel_dim;
        const float half = params_.sensor_range;
        const int   N    = static_cast<int>(2.0f * half / res) + 1;
        const int   W    = params_.neighbor_window;

        // dist_map[i][j] = 到最近障碍物的距离
        std::vector<std::vector<float>> dist_map(N, std::vector<float>(N, 1e6f));

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                const float x = odom_pos_.x - half + i * res;
                const float y = odom_pos_.y - half + j * res;

                // 查询最近障碍物距离
                PCLPoint query;
                query.x = x; query.y = y; query.z = odom_pos_.z;
                std::vector<int> k_idx(1);
                std::vector<float> k_dist(1);
                if (obs_kdtree_.nearestKSearch(query, 1, k_idx, k_dist) > 0) {
                    dist_map[i][j] = std::sqrt(k_dist[0]);
                }
            }
        }

        // 寻找局部最大值（在 [W x W] 窗口内距离最大）
        for (int i = W; i < N - W; i++) {
            for (int j = W; j < N - W; j++) {
                const float d = dist_map[i][j];
                if (d < params_.min_dist_to_obs) continue; // 太近障碍物

                bool is_local_max = true;
                for (int di = -W; di <= W && is_local_max; di++) {
                    for (int dj = -W; dj <= W && is_local_max; dj++) {
                        if (di == 0 && dj == 0) continue;
                        if (dist_map[i+di][j+dj] > d + 1e-4f) {
                            is_local_max = false;
                        }
                    }
                }

                if (is_local_max) {
                    const float x = odom_pos_.x - half + i * res;
                    const float y = odom_pos_.y - half + j * res;
                    // z 从地形获取（如果有 MapHandler）
                    float z = odom_pos_.z;
                    if (map_handler_ref_) {
                        bool is_matched = false;
                        z = MapHandler::NearestTerrainHeightofNavPoint(
                            Point3D(x, y, odom_pos_.z), is_matched);
                        if (!is_matched) z = odom_pos_.z;
                    }
                    candidates.emplace_back(x, y, z);
                }
            }
        }
    }

    /**
     * @brief 对候选点进行聚类去重
     *        遍历候选点列表，若与已接受点距离 < cluster_radius 则跳过
     */
    void ClusterFilter(const std::vector<Point3D>& in, std::vector<Point3D>& out) const {
        out.clear();
        const float r2 = params_.cluster_radius * params_.cluster_radius;
        for (const auto& p : in) {
            bool too_close = false;
            for (const auto& accepted : out) {
                const float dx = p.x - accepted.x;
                const float dy = p.y - accepted.y;
                if (dx*dx + dy*dy < r2) { too_close = true; break; }
            }
            if (!too_close) out.push_back(p);
        }
    }

    /**
     * @brief 将相邻骨架点连成边（用于可视化），并填充 global_contour_
     */
    void BuildSkeletonEdges(const std::vector<Point3D>& skeleton,
                            std::vector<PointStack>&    realworld_contour)
    {
        skeleton_edges_.clear();
        global_contour_.clear();
        if (skeleton.size() < 2) return;

        // 简单的近邻连接：每个点连接距离最近的点（但不超过 cluster_radius * 2）
        const float max_edge_dist = params_.cluster_radius * 3.0f;
        const float max_sq = max_edge_dist * max_edge_dist;

        for (std::size_t i = 0; i < skeleton.size(); i++) {
            float best_dist = max_sq;
            int   best_j    = -1;
            for (std::size_t j = i+1; j < skeleton.size(); j++) {
                const float dx = skeleton[i].x - skeleton[j].x;
                const float dy = skeleton[i].y - skeleton[j].y;
                const float d2 = dx*dx + dy*dy;
                if (d2 < best_dist) { best_dist = d2; best_j = static_cast<int>(j); }
            }
            if (best_j >= 0) {
                skeleton_edges_.emplace_back(skeleton[i], skeleton[best_j]);
                global_contour_.emplace_back(skeleton[i], skeleton[best_j]);
                // 也放进 realworld_contour 供 DynamicGraph 做连接判断
                PointStack edge_pts = {skeleton[i], skeleton[best_j]};
                realworld_contour.push_back(edge_pts);
            }
        }
    }
};
