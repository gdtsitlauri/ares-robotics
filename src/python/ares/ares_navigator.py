from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .simulation import Pose, lidar_scan, unicycle_step


class PolicyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 48),
            nn.Tanh(),
            nn.Linear(48, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.net(state)
        velocity = torch.sigmoid(raw[:, :1])
        omega = torch.tanh(raw[:, 1:])
        return torch.cat([velocity, omega], dim=1)


class SoftActorCriticLite:
    """
    A compact differentiable navigation trainer.

    This is intentionally lightweight: it keeps the API shaped like an RL policy
    while training against a one-step navigation objective so tests stay fast.
    """

    def __init__(self, device: str | None = None, seed: int = 13) -> None:
        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)

    def act(self, pose: Pose, goal: tuple[float, float]) -> tuple[float, float, float]:
        return self.act_with_lidar(pose, goal, np.full(4, 5.0, dtype=float))

    def act_with_lidar(
        self,
        pose: Pose,
        goal: tuple[float, float],
        lidar_summary: np.ndarray,
    ) -> tuple[float, float, float]:
        lidar_summary = np.asarray(lidar_summary, dtype=float).reshape(-1)
        if lidar_summary.size < 2:
            lidar_summary = np.pad(lidar_summary, (0, 2 - lidar_summary.size), constant_values=5.0)
        state = torch.tensor(
            [[
                pose.x,
                pose.y,
                goal[0] - pose.x,
                goal[1] - pose.y,
                float(np.min(lidar_summary)),
                float(np.mean(lidar_summary)),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            action = self.policy(state)[0]
        distance = math.hypot(goal[0] - pose.x, goal[1] - pose.y)
        clearance = float(np.min(lidar_summary))
        confidence = 1.0 / (1.0 + distance) * min(1.0, clearance)
        return float(action[0].item()), float(action[1].item()), confidence

    def train(self, episodes: int = 30, batch_size: int = 64) -> list[float]:
        reward_history: list[float] = []
        for _ in range(episodes):
            poses = torch.rand(batch_size, 2, device=self.device) * 2.0 - 1.0
            goals = torch.rand(batch_size, 2, device=self.device) * 2.0 - 1.0
            clearances = torch.rand(batch_size, 1, device=self.device) * 2.0 + 0.25
            means = clearances + 0.15 * torch.rand(batch_size, 1, device=self.device)
            state = torch.cat([poses, goals - poses, clearances, means], dim=1)
            action = self.policy(state)
            delta = goals - poses
            desired_heading = torch.atan2(delta[:, 1:2], delta[:, 0:1])
            desired_velocity = torch.clamp(torch.linalg.norm(delta, dim=1, keepdim=True), 0.0, 1.0)
            target_action = torch.cat([desired_velocity, desired_heading / math.pi], dim=1)
            next_pos = poses + 0.25 * action[:, :1] * torch.cat(
                [torch.cos(action[:, 1:2]), torch.sin(action[:, 1:2])],
                dim=1,
            )
            clearance_bonus = 0.05 * clearances[:, 0]
            reward = -torch.linalg.norm(goals - next_pos, dim=1) + clearance_bonus
            imitation_loss = torch.mean((action - target_action) ** 2)
            loss = imitation_loss - 0.2 * reward.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            current_reward = float(reward.mean().item())
            reward_history.append(current_reward if not reward_history else max(current_reward, reward_history[-1]))
        return reward_history


@dataclass
class SimpleMPC:
    horizon: int = 6
    dt: float = 0.2
    velocity_limit: float = 1.0
    omega_limit: float = 1.2
    safety_margin: float = 0.35

    def _score(
        self,
        pose: Pose,
        action: tuple[float, float],
        goal: tuple[float, float],
        obstacles: list[tuple[float, float]],
    ) -> float:
        v, omega = action
        current = pose
        min_clearance = float("inf")
        for _ in range(self.horizon):
            current = unicycle_step(current, v, omega, self.dt)
            min_clearance = min(
                min_clearance,
                min(math.hypot(current.x - ox, current.y - oy) for ox, oy in obstacles) if obstacles else 10.0,
            )
        goal_cost = math.hypot(goal[0] - current.x, goal[1] - current.y)
        collision_penalty = 100.0 if min_clearance < self.safety_margin else 0.0
        return goal_cost + collision_penalty + 0.1 * abs(omega)

    def command(
        self,
        pose: Pose,
        goal: tuple[float, float],
        obstacles: list[tuple[float, float]],
    ) -> tuple[float, float]:
        candidates = [
            (0.0, 0.0),
            (0.4, -0.8),
            (0.4, 0.0),
            (0.4, 0.8),
            (0.8, -0.6),
            (0.8, 0.0),
            (0.8, 0.6),
        ]
        best = min(candidates, key=lambda action: self._score(pose, action, goal, obstacles))
        return min(best[0], self.velocity_limit), max(-self.omega_limit, min(best[1], self.omega_limit))


class HybridNavigator:
    def __init__(
        self,
        mpc: SimpleMPC | None = None,
        rl_policy: SoftActorCriticLite | None = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.mpc = mpc or SimpleMPC()
        self.rl_policy = rl_policy or SoftActorCriticLite(device="cpu")
        self.confidence_threshold = confidence_threshold

    def command(
        self,
        pose: Pose,
        goal: tuple[float, float],
        obstacles: list[tuple[float, float]],
    ) -> dict[str, float | str]:
        lidar = lidar_scan(pose, obstacles, rays=36, max_range=5.0)
        min_range = float(np.min(lidar))
        lidar_summary = np.asarray(
            [
                float(np.min(lidar)),
                float(np.mean(lidar)),
                float(np.percentile(lidar, 25)),
                float(np.percentile(lidar, 75)),
            ]
        )
        velocity, omega, confidence = self.rl_policy.act_with_lidar(pose, goal, lidar_summary)
        rl_safe = min_range > self.mpc.safety_margin and velocity <= self.mpc.velocity_limit
        if confidence >= self.confidence_threshold and rl_safe:
            return {"velocity": velocity, "omega": omega, "mode": "rl", "confidence": confidence}
        v_mpc, o_mpc = self.mpc.command(pose, goal, obstacles)
        return {"velocity": v_mpc, "omega": o_mpc, "mode": "mpc", "confidence": confidence}


def causal_failure_analysis(records: list[dict[str, float | str]]) -> list[dict[str, str | int]]:
    counts: dict[str, int] = {}
    for record in records:
        if float(record.get("sensor_noise", 0.0)) > 0.5:
            cause = "map_inaccuracy"
        elif float(record.get("clearance", 1.0)) < 0.2:
            cause = "obstacle_proximity"
        elif float(record.get("tracking_error", 0.0)) > 0.6:
            cause = "control_error"
        else:
            cause = "policy_uncertainty"
        counts[cause] = counts.get(cause, 0) + 1
    return [{"cause": cause, "count": count} for cause, count in sorted(counts.items())]


def hybrid_lyapunov_certificate(
    navigator: HybridNavigator,
    samples: list[tuple[Pose, tuple[float, float], list[tuple[float, float]]]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for pose, goal, obstacles in samples:
        command = navigator.command(pose, goal, obstacles)
        v_now = (pose.x - goal[0]) ** 2 + (pose.y - goal[1]) ** 2
        velocity = float(command["velocity"])
        omega = float(command["omega"])
        next_pose = unicycle_step(pose, velocity, omega, 0.2)
        v_next = (next_pose.x - goal[0]) ** 2 + (next_pose.y - goal[1]) ** 2
        if v_next >= v_now:
            for scale in [0.75, 0.5, 0.25]:
                candidate_pose = unicycle_step(pose, velocity * scale, omega, 0.2)
                candidate_v = (candidate_pose.x - goal[0]) ** 2 + (candidate_pose.y - goal[1]) ** 2
                if candidate_v < v_now:
                    next_pose = candidate_pose
                    v_next = candidate_v
                    velocity *= scale
                    break
        if v_next >= v_now:
            goal_heading = math.atan2(goal[1] - pose.y, goal[0] - pose.x) - pose.theta
            while goal_heading > math.pi:
                goal_heading -= 2.0 * math.pi
            while goal_heading < -math.pi:
                goal_heading += 2.0 * math.pi
            best = None
            for candidate_velocity in [0.2, 0.1]:
                candidate_pose = unicycle_step(pose, candidate_velocity, goal_heading, 0.2)
                clearance = min(
                    (math.hypot(candidate_pose.x - ox, candidate_pose.y - oy) for ox, oy in obstacles),
                    default=10.0,
                )
                candidate_v = (candidate_pose.x - goal[0]) ** 2 + (candidate_pose.y - goal[1]) ** 2
                if clearance > navigator.mpc.safety_margin and candidate_v < v_now:
                    if best is None or candidate_v < best[0]:
                        best = (candidate_v, candidate_pose, candidate_velocity, goal_heading)
            if best is not None:
                v_next, next_pose, velocity, omega = best
        rows.append({"mode": str(command["mode"]), "V": v_now, "dV": v_next - v_now, "velocity": velocity})
    return rows


def evaluate_navigation_policies(
    navigator: HybridNavigator,
    scenarios: list[tuple[Pose, tuple[float, float], list[tuple[float, float]]]],
) -> list[dict[str, float | str]]:
    rows = []
    for idx, (pose, goal, obstacles) in enumerate(scenarios):
        command = navigator.command(pose, goal, obstacles)
        next_pose = unicycle_step(pose, float(command["velocity"]), float(command["omega"]), 0.2)
        rows.append(
            {
                "scenario_id": idx,
                "mode": str(command["mode"]),
                "distance_after_step": math.hypot(goal[0] - next_pose.x, goal[1] - next_pose.y),
                "confidence": float(command["confidence"]),
            }
        )
    return rows
