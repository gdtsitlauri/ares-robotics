from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    x: float
    y: float
    theta: float


def bicycle_step(pose: Pose, velocity: float, steering: float, wheelbase: float, dt: float) -> Pose:
    theta_dot = velocity * math.tan(steering) / max(wheelbase, 1e-9)
    return Pose(
        x=pose.x + velocity * math.cos(pose.theta) * dt,
        y=pose.y + velocity * math.sin(pose.theta) * dt,
        theta=pose.theta + theta_dot * dt,
    )


def differential_drive_step(pose: Pose, left: float, right: float, track_width: float, dt: float) -> Pose:
    velocity = 0.5 * (left + right)
    omega = (right - left) / max(track_width, 1e-9)
    return Pose(
        x=pose.x + velocity * math.cos(pose.theta) * dt,
        y=pose.y + velocity * math.sin(pose.theta) * dt,
        theta=pose.theta + omega * dt,
    )


def unicycle_step(pose: Pose, velocity: float, omega: float, dt: float) -> Pose:
    return Pose(
        x=pose.x + velocity * math.cos(pose.theta) * dt,
        y=pose.y + velocity * math.sin(pose.theta) * dt,
        theta=pose.theta + omega * dt,
    )


def lidar_scan(
    pose: Pose,
    obstacles: list[tuple[float, float]],
    rays: int = 36,
    max_range: float = 10.0,
) -> np.ndarray:
    readings = np.full(rays, max_range, dtype=float)
    for idx in range(rays):
        angle = pose.theta + (2.0 * math.pi * idx / rays)
        ray_dir = np.array([math.cos(angle), math.sin(angle)])
        origin = np.array([pose.x, pose.y])
        for obstacle in obstacles:
            delta = np.asarray(obstacle, dtype=float) - origin
            projection = float(delta @ ray_dir)
            if projection <= 0 or projection > max_range:
                continue
            lateral = np.linalg.norm(delta - projection * ray_dir)
            if lateral < 0.25:
                readings[idx] = min(readings[idx], projection)
    return readings


def gps_measurement(pose: Pose, sigma: float = 0.05, seed: int = 3) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=2)
    return pose.x + float(noise[0]), pose.y + float(noise[1])


def imu_measurement(
    linear_accel: float,
    angular_rate: float,
    accel_sigma: float = 0.02,
    gyro_sigma: float = 0.01,
    seed: int = 5,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    accel = linear_accel + float(rng.normal(0.0, accel_sigma))
    gyro = angular_rate + float(rng.normal(0.0, gyro_sigma))
    return accel, gyro


def camera_features(
    pose: Pose,
    landmarks: list[tuple[float, float]],
    fov: float = math.pi / 2.0,
    max_range: float = 8.0,
) -> list[dict[str, float]]:
    features: list[dict[str, float]] = []
    for x, y in landmarks:
        dx = x - pose.x
        dy = y - pose.y
        distance = math.hypot(dx, dy)
        if distance > max_range:
            continue
        bearing = math.atan2(dy, dx) - pose.theta
        while bearing > math.pi:
            bearing -= 2.0 * math.pi
        while bearing < -math.pi:
            bearing += 2.0 * math.pi
        if abs(bearing) <= fov / 2.0:
            features.append({"x": x, "y": y, "distance": distance, "bearing": bearing})
    return features


def scenario_obstacles(name: str, step: int = 0) -> list[tuple[float, float]]:
    if name == "maze_navigation":
        return [(1.0, 0.5), (1.5, -0.3), (2.0, 0.4), (2.5, -0.2)]
    if name == "dynamic_obstacles":
        return [(1.2 + 0.02 * step, 0.1), (1.8, -0.4 + 0.015 * step)]
    if name == "multi_robot":
        return [(1.4, 0.2), (2.0, 0.0), (2.6, -0.2)]
    if name == "narrow_passage":
        return [(1.4, 0.35), (1.4, -0.35), (2.0, 0.35), (2.0, -0.35)]
    return []


def simulate_navigation_episode(
    navigator,
    scenario: str,
    goal: tuple[float, float] = (3.0, 0.0),
    steps: int = 40,
    dt: float = 0.2,
) -> dict[str, float | bool | str]:
    pose = Pose(0.0, 0.0, 0.0)
    path = [(pose.x, pose.y)]
    collision = False
    mode_counts = {"rl": 0, "mpc": 0}

    for step in range(steps):
        obstacles = scenario_obstacles(scenario, step)
        command = navigator.command(pose, goal, obstacles)
        mode_counts[str(command["mode"])] = mode_counts.get(str(command["mode"]), 0) + 1
        pose = unicycle_step(pose, float(command["velocity"]), float(command["omega"]), dt)
        path.append((pose.x, pose.y))
        min_clearance = min(
            (math.hypot(pose.x - ox, pose.y - oy) for ox, oy in obstacles),
            default=10.0,
        )
        if min_clearance < 0.22:
            collision = True
            break
        if math.hypot(goal[0] - pose.x, goal[1] - pose.y) < 0.35:
            break

    travelled = sum(math.dist(path[idx - 1], path[idx]) for idx in range(1, len(path)))
    success = not collision and math.hypot(goal[0] - pose.x, goal[1] - pose.y) < 0.5
    return {
        "scenario": scenario,
        "success": success,
        "path_length": travelled,
        "final_distance": math.hypot(goal[0] - pose.x, goal[1] - pose.y),
        "planning_time_ms": steps * 0.25,
        "rl_steps": mode_counts.get("rl", 0),
        "mpc_steps": mode_counts.get("mpc", 0),
    }
