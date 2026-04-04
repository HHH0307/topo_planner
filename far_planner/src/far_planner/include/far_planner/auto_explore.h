/**
 * @file auto_explore.h
 * @author hqy
 * @brief 自主探索头文件
 * @date 2026-04-01
 */

#pragma once

#include "utility.h"

struct AutoExploreParams {
	AutoExploreParams() = default;
	float min_frontier_dist = 2.0f;
	float min_goal_separation = 1.0f;
};

class AutoExplore {
public:
	AutoExplore() = default;
	~AutoExplore() = default;

	inline void Init(const AutoExploreParams& params) {
		params_ = params;
		this->Reset();
	}

	inline void Reset() {
		has_last_goal_ = false;
		is_completed_ = false;
		has_active_goal_ = false;
		last_goal_pos_ = Point3D(0, 0, 0);
	}

	inline bool IsCompleted() const { return is_completed_; }
	inline bool HasActiveGoal() const { return has_active_goal_; }
	inline void ClearActiveGoal() { has_active_goal_ = false; }
	inline void MarkGoalSelected(const Point3D& goal) {
		has_last_goal_ = true;
		has_active_goal_ = true;
		last_goal_pos_ = goal;
	}

	bool SelectGoalFromFrontier(const NodePtrStack& nav_graph,
								const NavNodePtr& odom_node_ptr,
								Point3D& goal_out);

private:
	bool SelectCandidate(const NodePtrStack& nav_graph,
						 const NavNodePtr& odom_node_ptr,
						 const bool avoid_last_goal,
						 Point3D& goal_out);

	AutoExploreParams params_;
	bool has_last_goal_ = false;
	bool has_active_goal_ = false;
	bool is_completed_ = false;
	Point3D last_goal_pos_ = Point3D(0, 0, 0);
};

