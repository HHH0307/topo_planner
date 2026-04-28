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
	// 到达目标的收敛距离，到达后将目标加入已访问黑名单
	float goal_reached_dist = 1.5f;
	// 单个目标的最大尝试时间（秒），超时则切换目标
	float goal_timeout_sec = 30.0f;
	// 已访问目标的记忆窗口大小
	int   visited_memory_size = 10;
	// 信息增益权重（邻近前沿数量）
	float info_gain_weight = 1.0f;
	// 路径代价权重
	float path_cost_weight = 0.5f;
	// 方向多样性权重（鼓励向未探索方向前进）
	float direction_weight = 0.3f;
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
		visited_goals_.clear();
		goal_start_time_ = std::chrono::steady_clock::now();
		explore_start_pos_ = Point3D(0, 0, 0);
		has_explore_start_ = false;
	}

	inline bool IsCompleted() const { return is_completed_; }
	inline bool HasActiveGoal() const { return has_active_goal_; }
	inline Point3D GetLastGoalPos() const { return last_goal_pos_; }
	inline void ClearActiveGoal() { has_active_goal_ = false; }

	inline void MarkGoalSelected(const Point3D& goal) {
		has_last_goal_ = true;
		has_active_goal_ = true;
		last_goal_pos_ = goal;
		goal_start_time_ = std::chrono::steady_clock::now();
	}

	/**
	 * @brief 检查当前目标是否超时，超时则放弃并加入黑名单
	 * @param robot_pos 机器人当前位置
	 * @return true if goal was abandoned due to timeout
	 */
	inline bool CheckAndHandleGoalTimeout(const Point3D& robot_pos) {
		if (!has_active_goal_) return false;
		const float elapsed = std::chrono::duration<float>(
			std::chrono::steady_clock::now() - goal_start_time_).count();
		if (elapsed > params_.goal_timeout_sec) {
			// 将超时目标加入已访问黑名单
			AddVisitedGoal(last_goal_pos_);
			has_active_goal_ = false;
			return true;
		}
		return false;
	}

	/**
	 * @brief 检查是否已到达当前目标（到达后加入黑名单，允许选新目标）
	 */
	inline bool CheckGoalReached(const Point3D& robot_pos) {
		if (!has_active_goal_) return false;
		if ((robot_pos - last_goal_pos_).norm_flat() < params_.goal_reached_dist) {
			AddVisitedGoal(last_goal_pos_);
			has_active_goal_ = false;
			return true;
		}
		return false;
	}

	bool SelectGoalFromFrontier(const NodePtrStack& nav_graph,
								const NavNodePtr& odom_node_ptr,
								Point3D& goal_out);

private:
	/**
	 * @brief 计算某个前沿节点的综合效用分数
	 *   = info_gain_weight * 邻近前沿数
	 *   + path_cost_weight * path_accessibility
	 *   + direction_weight * 方向多样性得分
	 */
	float ComputeUtility(const NavNodePtr& node_ptr,
						 const NavNodePtr& odom_node_ptr,
						 const NodePtrStack& nav_graph) const;

	bool SelectCandidate(const NodePtrStack& nav_graph,
						 const NavNodePtr& odom_node_ptr,
						 const bool avoid_visited,
						 Point3D& goal_out);

	inline void AddVisitedGoal(const Point3D& pos) {
		visited_goals_.push_back(pos);
		if ((int)visited_goals_.size() > params_.visited_memory_size) {
			visited_goals_.pop_front();
		}
	}

	inline bool IsNearVisitedGoal(const Point3D& pos) const {
		for (const auto& vg : visited_goals_) {
			if ((pos - vg).norm_flat() < params_.min_goal_separation) return true;
		}
		return false;
	}

	AutoExploreParams params_;
	bool has_last_goal_ = false;
	bool has_active_goal_ = false;
	bool is_completed_ = false;
	Point3D last_goal_pos_ = Point3D(0, 0, 0);

	// 已访问目标黑名单（滑动窗口）
	std::deque<Point3D> visited_goals_;

	// 超时检测
	std::chrono::steady_clock::time_point goal_start_time_;

	// 探索起始位置，用于计算方向多样性
	Point3D explore_start_pos_;
	bool has_explore_start_ = false;
};
