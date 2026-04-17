#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef ARES_WITH_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace {

using Node = std::pair<int, int>;
using Vec2 = std::pair<double, double>;

struct NodeHash {
  std::size_t operator()(const Node& node) const noexcept {
    return static_cast<std::size_t>((node.first << 16) ^ node.second);
  }
};

struct QueueItem {
  double priority;
  Node node;

  bool operator<(const QueueItem& other) const { return priority > other.priority; }
};

double heuristic(const Node& a, const Node& b) {
  return std::abs(a.first - b.first) + std::abs(a.second - b.second);
}

std::vector<Node> astar_cpp(
    int width,
    int height,
    const std::unordered_set<Node, NodeHash>& obstacles,
    const Node& start,
    const Node& goal) {
  auto in_bounds = [&](const Node& node) {
    return node.first >= 0 && node.first < width && node.second >= 0 && node.second < height;
  };
  auto passable = [&](const Node& node) { return obstacles.find(node) == obstacles.end(); };

  std::priority_queue<QueueItem> frontier;
  frontier.push({0.0, start});
  std::unordered_map<Node, Node, NodeHash> came_from;
  std::unordered_map<Node, double, NodeHash> cost;
  came_from[start] = start;
  cost[start] = 0.0;

  const std::vector<Node> deltas = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  while (!frontier.empty()) {
    auto current = frontier.top().node;
    frontier.pop();
    if (current == goal) {
      std::vector<Node> path{goal};
      Node cursor = goal;
      while (cursor != start) {
        cursor = came_from.at(cursor);
        path.push_back(cursor);
      }
      std::reverse(path.begin(), path.end());
      return path;
    }

    for (const auto& delta : deltas) {
      Node next{current.first + delta.first, current.second + delta.second};
      if (!in_bounds(next) || !passable(next)) {
        continue;
      }
      double new_cost = cost[current] + 1.0;
      if (!cost.count(next) || new_cost < cost[next]) {
        cost[next] = new_cost;
        came_from[next] = current;
        frontier.push({new_cost + heuristic(next, goal), next});
      }
    }
  }
  throw std::runtime_error("No path found");
}

Vec2 simulate_unicycle_step(const Vec2& position, double heading, double velocity, double omega, double dt) {
  return {position.first + velocity * std::cos(heading) * dt,
          position.second + velocity * std::sin(heading) * dt};
}

bool collision_free_path(const std::vector<Vec2>& path, const std::vector<Vec2>& obstacles, double radius) {
  for (const auto& point : path) {
    for (const auto& obstacle : obstacles) {
      if (std::hypot(point.first - obstacle.first, point.second - obstacle.second) < radius) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

#ifdef ARES_WITH_PYBIND
namespace py = pybind11;

PYBIND11_MODULE(planning_core, m) {
  m.doc() = "ARES C++ planning core";
  m.def(
      "astar_cpp",
      [](int width, int height, const std::vector<Node>& obstacles, Node start, Node goal) {
        return astar_cpp(width, height, {obstacles.begin(), obstacles.end()}, start, goal);
      },
      py::arg("width"),
      py::arg("height"),
      py::arg("obstacles"),
      py::arg("start"),
      py::arg("goal"));
  m.def(
      "simulate_unicycle_step",
      &simulate_unicycle_step,
      py::arg("position"),
      py::arg("heading"),
      py::arg("velocity"),
      py::arg("omega"),
      py::arg("dt"));
  m.def(
      "collision_free_path",
      &collision_free_path,
      py::arg("path"),
      py::arg("obstacles"),
      py::arg("radius"));
}
#endif
