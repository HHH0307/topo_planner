/**
 * @file auto_explore.cpp
 * @author hqy
 * @brief 自主探索
 * @date 2026-04-01
 */

#include "far_planner/auto_explore.h"


bool AutoExplore::SelectCandidate(const NodePtrStack& nav_graph,
								  const NavNodePtr& odom_node_ptr,
								  const bool avoid_last_goal,
								  Point3D& goal_out)
{
	if (odom_node_ptr == NULL || nav_graph.empty()) return false;

	float best_score = -1.0f;
	NavNodePtr best_ptr = NULL;
	for (const auto& node_ptr : nav_graph) {
		if (node_ptr == NULL || node_ptr->is_odom || !node_ptr->is_frontier || !node_ptr->is_traversable) continue;

		const float dist_to_robot = (node_ptr->position - odom_node_ptr->position).norm_flat();
		if (dist_to_robot < params_.min_frontier_dist) continue;
		if (avoid_last_goal && has_last_goal_ && (node_ptr->position - last_goal_pos_).norm_flat() < params_.min_goal_separation) continue;

		// 优先选择距离机器人更远的可到达点位。
		float score = node_ptr->gscore;
		if (score >= FARUtil::kINF / 2.0f) {
			score = dist_to_robot;
		}
		if (score > best_score) {
			best_score = score;
			best_ptr = node_ptr;
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
	if (is_completed_) return false;
	// 尽量避免选取与上一个目标距离过近的点位；若无可用候选目标，则放宽筛选约束。
	if (!this->SelectCandidate(nav_graph, odom_node_ptr, true, goal_out)) {
		if (!this->SelectCandidate(nav_graph, odom_node_ptr, false, goal_out)) {
			is_completed_ = true;
			has_active_goal_ = false;
			return false;
		}
	}
	this->MarkGoalSelected(goal_out);
	is_completed_ = false;
	return true;
}



