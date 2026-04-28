/**
 * @file auto_explore.cpp
 * @author hqy
 * @brief 自主探索 —— 优化版
 *
 * 改进说明（相比原版）：
 * 1. 多目标效用函数：综合「信息增益」「路径可达性」「方向多样性」三项打分，
 *    替代原来只看 gscore 的单一排序。
 * 2. 已访问目标黑名单（滑动窗口）：避免来回重复访问同一前沿节点。
 * 3. 目标超时放弃：若在 goal_timeout_sec 内未到达，主动放弃当前目标并切换，
 *    防止机器人长时间卡死在一个地方。
 * 4. 到达检测放入 AutoExplore 内部：到达后自动将目标加入黑名单并触发下一次选取。
 * 5. is_completed_ 每轮重置：只有当轮次确实找不到任何前沿时才标记完成，
 *    避免因一次失败而永久停止探索。
 */

#include "far_planner/auto_explore.h"

float AutoExplore::ComputeUtility(const NavNodePtr& node_ptr,
                                  const NavNodePtr& odom_node_ptr,
                                  const NodePtrStack& nav_graph) const
{
    // ── 1. 信息增益：邻近前沿邻居数量（反映局部未知区域丰富程度）
    int frontier_neighbor_count = 0;
    for (const auto& nb : node_ptr->connect_nodes) {
        if (nb != NULL && nb->is_frontier) {
            frontier_neighbor_count++;
        }
    }
    // 再加上轮廓连接中的前沿节点
    for (const auto& nb : node_ptr->contour_connects) {
        if (nb != NULL && nb->is_frontier) {
            frontier_neighbor_count++;
        }
    }
    const float info_gain = static_cast<float>(frontier_neighbor_count);

    // ── 2. 路径可达性得分：路径代价越短（越容易到达）得分越高
    //   gscore < kINF/2 说明已经有有效 A* 路径
    float path_score = 0.0f;
    const float dist = (node_ptr->position - odom_node_ptr->position).norm_flat();
    if (node_ptr->gscore < FARUtil::kINF / 2.0f && node_ptr->gscore > FARUtil::kEpsilon) {
        // 距离适中时效用最高：太近没意义，太远路途难以保证
        // 用高斯形状对距离进行加权，峰值在 sensor_range 的一半处
        const float ideal_dist = std::max(params_.min_frontier_dist * 3.0f, 5.0f);
        const float sigma = ideal_dist * 0.8f;
        path_score = std::exp(-0.5f * std::pow((dist - ideal_dist) / sigma, 2.0f));
    } else {
        // 无可达路径，给一个很小的分数（仍可被兜底策略选中）
        path_score = 0.05f;
    }

    // ── 3. 方向多样性：优先选择与上一目标方向不同的前沿（减少来回振荡）
    float dir_score = 1.0f;
    if (has_last_goal_ && has_explore_start_) {
        const Point3D last_dir = (last_goal_pos_ - explore_start_pos_).normalize();
        const Point3D new_dir  = (node_ptr->position - odom_node_ptr->position).normalize();
        // dot product in flat plane
        const float dot = last_dir.x * new_dir.x + last_dir.y * new_dir.y;
        // [-1,1] -> [0,1], 值越小代表方向差异越大
        dir_score = (1.0f - dot) / 2.0f + 0.1f; // [0.1, 1.1]
    }

    return params_.info_gain_weight  * info_gain
         + params_.path_cost_weight  * path_score
         + params_.direction_weight  * dir_score;
}

bool AutoExplore::SelectCandidate(const NodePtrStack& nav_graph,
                                  const NavNodePtr& odom_node_ptr,
                                  const bool avoid_visited,
                                  Point3D& goal_out)
{
    if (odom_node_ptr == NULL || nav_graph.empty()) return false;

    float best_score = -1.0f;
    NavNodePtr best_ptr = NULL;

    for (const auto& node_ptr : nav_graph) {
        if (node_ptr == NULL || node_ptr->is_odom) continue;
        if (!node_ptr->is_frontier || !node_ptr->is_traversable) continue;

        const float dist = (node_ptr->position - odom_node_ptr->position).norm_flat();
        if (dist < params_.min_frontier_dist) continue;

        // 已访问黑名单过滤
        if (avoid_visited && IsNearVisitedGoal(node_ptr->position)) continue;

        const float score = ComputeUtility(node_ptr, odom_node_ptr, nav_graph);
        if (score > best_score) {
            best_score = score;
            best_ptr   = node_ptr;
        }
    }

    if (best_ptr == NULL) return false;
    goal_out = best_ptr->position;
    return true;
}

bool AutoExplore::SelectGoalFromFrontier(const NodePtrStack& nav_graph,
                                         const NavNodePtr& odom_node_ptr,
                                         Point3D& goal_out)
{
    // 每次调用前重置完成状态，只有本轮真的找不到才设 true
    is_completed_ = false;

    if (!has_explore_start_ && odom_node_ptr != NULL) {
        explore_start_pos_ = odom_node_ptr->position;
        has_explore_start_ = true;
    }

    // 优先：避开已访问目标选最优前沿
    if (SelectCandidate(nav_graph, odom_node_ptr, true, goal_out)) {
        MarkGoalSelected(goal_out);
        // 每次选新目标时更新起始位置，用于下一次方向多样性计算
        if (odom_node_ptr != NULL) {
            explore_start_pos_ = odom_node_ptr->position;
        }
        return true;
    }

    // 兜底：放宽约束，忽略黑名单
    if (SelectCandidate(nav_graph, odom_node_ptr, false, goal_out)) {
        MarkGoalSelected(goal_out);
        if (odom_node_ptr != NULL) {
            explore_start_pos_ = odom_node_ptr->position;
        }
        return true;
    }

    // 本轮确实找不到任何前沿
    is_completed_ = true;
    has_active_goal_ = false;
    return false;
}
