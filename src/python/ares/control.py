from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


@dataclass
class FractionalPIDGains:
    kp: float
    ki: float
    kd: float
    alpha: float = 0.9
    beta: float = 0.7


class PIDController:
    """Discrete PID controller with simple anti-windup clamping."""

    def __init__(
        self,
        gains: PIDGains,
        integral_limit: float = 10.0,
        adaptive: bool = False,
    ) -> None:
        self.gains = gains
        self.integral_limit = integral_limit
        self.adaptive = adaptive
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def tune(self, error: float) -> None:
        if not self.adaptive:
            return
        scale = min(1.0, abs(error))
        self.gains.kp = max(0.1, self.gains.kp * (1.0 + 0.01 * scale))
        self.gains.kd = max(0.0, self.gains.kd * (1.0 + 0.005 * scale))

    def step(self, error: float, dt: float) -> float:
        self.tune(error)
        self.integral = np.clip(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        derivative = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error
        return (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * derivative
        )


class CascadePIDController:
    """Outer loop regulates the reference for an inner loop."""

    def __init__(self, outer: PIDController, inner: PIDController) -> None:
        self.outer = outer
        self.inner = inner

    def reset(self) -> None:
        self.outer.reset()
        self.inner.reset()

    def step(
        self,
        outer_error: float,
        measured_inner_state: float,
        dt: float,
    ) -> float:
        inner_target = self.outer.step(outer_error, dt)
        inner_error = inner_target - measured_inner_state
        return self.inner.step(inner_error, dt)


class FractionalPIDController:
    """
    Truncated-memory fractional-order PI^a D^b controller.

    The implementation uses power-law weights over a short error history to
    approximate fractional integration and differentiation while remaining fast.
    """

    def __init__(self, gains: FractionalPIDGains, memory: int = 32) -> None:
        self.gains = gains
        self.memory = memory
        self.errors: list[float] = []

    def reset(self) -> None:
        self.errors.clear()

    def _weighted_sum(self, exponent: float) -> float:
        total = 0.0
        for idx, err in enumerate(reversed(self.errors), start=1):
            total += err / (idx ** exponent)
        return total

    def step(self, error: float, dt: float) -> float:
        self.errors.append(error)
        if len(self.errors) > self.memory:
            self.errors.pop(0)
        frac_integral = dt ** self.gains.alpha * self._weighted_sum(1.0 - self.gains.alpha)
        frac_derivative = 0.0
        if len(self.errors) > 1:
            diffs = [self.errors[idx] - self.errors[idx - 1] for idx in range(1, len(self.errors))]
            for idx, diff in enumerate(reversed(diffs), start=1):
                frac_derivative += diff / (idx ** (1.0 - self.gains.beta))
            frac_derivative /= max(dt ** self.gains.beta, 1e-9)
        return (
            self.gains.kp * error
            + self.gains.ki * frac_integral
            + self.gains.kd * frac_derivative
        )


def ziegler_nichols(ultimate_gain: float, oscillation_period: float) -> PIDGains:
    return PIDGains(
        kp=0.6 * ultimate_gain,
        ki=1.2 * ultimate_gain / oscillation_period,
        kd=0.075 * ultimate_gain * oscillation_period,
    )


def cohen_coon(process_gain: float, delay: float, time_constant: float) -> PIDGains:
    ratio = delay / max(time_constant, 1e-9)
    kp = (1.35 / max(process_gain, 1e-9)) * (1.0 + ratio / 5.0)
    ki = kp / max(delay * (32.0 + 6.0 * ratio) / (13.0 + 8.0 * ratio), 1e-9)
    kd = kp * delay * 4.0 / (11.0 + 2.0 * ratio)
    return PIDGains(kp=kp, ki=ki, kd=kd)


def itae_tuning(gain: float, delay: float, time_constant: float) -> PIDGains:
    ratio = delay / max(time_constant, 1e-9)
    return PIDGains(
        kp=0.586 / max(gain, 1e-9) * ratio ** (-0.916),
        ki=1.03 / max(delay, 1e-9) * ratio ** (-0.165),
        kd=0.308 * delay * ratio ** (-0.929),
    )


def simulate_first_order_response(
    controller: PIDController,
    target: float = 1.0,
    dt: float = 0.02,
    steps: int = 250,
    plant_a: float = 1.1,
    plant_b: float = 1.0,
) -> list[dict[str, float]]:
    """Simulate x_dot = -a x + b u and return a table-friendly trace."""

    x = 0.0
    trace: list[dict[str, float]] = []
    for step in range(steps):
        error = target - x
        u = controller.step(error, dt)
        x_dot = -plant_a * x + plant_b * u
        x += dt * x_dot
        trace.append(
            {
                "time": step * dt,
                "target": target,
                "state": x,
                "error": error,
                "control": u,
            }
        )
    return trace


def controllability_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    return np.hstack([np.linalg.matrix_power(a, i) @ b for i in range(n)])


def observability_matrix(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    return np.vstack([c @ np.linalg.matrix_power(a, i) for i in range(n)])


def luenberger_observer_step(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    l: np.ndarray,
    x_hat: np.ndarray,
    control: np.ndarray,
    measurement: np.ndarray,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    l = np.asarray(l, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float).reshape(-1, 1)
    control = np.asarray(control, dtype=float).reshape(-1, 1)
    measurement = np.asarray(measurement, dtype=float).reshape(-1, 1)
    innovation = measurement - c @ x_hat
    return a @ x_hat + b @ control + l @ innovation


def kalman_filter_step(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    x_hat: np.ndarray,
    covariance: np.ndarray,
    control: np.ndarray,
    measurement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float).reshape(-1, 1)
    covariance = np.asarray(covariance, dtype=float)
    control = np.asarray(control, dtype=float).reshape(-1, 1)
    measurement = np.asarray(measurement, dtype=float).reshape(-1, 1)

    x_pred = a @ x_hat + b @ control
    p_pred = a @ covariance @ a.T + q
    s = c @ p_pred @ c.T + r
    kalman_gain = p_pred @ c.T @ np.linalg.inv(s)
    innovation = measurement - c @ x_pred
    x_next = x_pred + kalman_gain @ innovation
    identity = np.eye(covariance.shape[0])
    p_next = (identity - kalman_gain @ c) @ p_pred
    return x_next, p_next


def ackermann_pole_placement(
    a: np.ndarray,
    b: np.ndarray,
    desired_poles: Iterable[float],
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]
    ctrb = controllability_matrix(a, b)
    if np.linalg.matrix_rank(ctrb) != n:
        raise ValueError("System is not controllable")
    coeffs = np.poly(list(desired_poles))
    phi = np.zeros_like(a)
    for i, coeff in enumerate(coeffs):
        power = n - i
        term = coeff * (np.eye(n) if power == 0 else np.linalg.matrix_power(a, power))
        phi += term
    selector = np.zeros((1, n))
    selector[0, -1] = 1.0
    return selector @ np.linalg.inv(ctrb) @ phi


def lqr(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a discrete-time LQR problem using Riccati iteration."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    p = q.copy()
    for _ in range(iterations):
        bt_p = b.T @ p
        gain_term = np.linalg.inv(r + bt_p @ b)
        p = q + a.T @ (p - p @ b @ gain_term @ bt_p) @ a
    k = np.linalg.inv(r + b.T @ p @ b) @ (b.T @ p @ a)
    return k, p


def h_infinity_state_feedback(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    gamma: float = 1.4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Practical H-infinity surrogate based on disturbance-inflated state penalty.

    This keeps the implementation dependency-free while biasing the controller
    toward robustness against worst-case perturbations.
    """

    robust_q = np.asarray(q, dtype=float) * (gamma ** 2)
    robust_r = np.asarray(r, dtype=float) + np.eye(np.asarray(r).shape[0]) / max(gamma, 1e-9)
    return lqr(a, b, robust_q, robust_r)


def mu_synthesis_surrogate(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    uncertainty_levels: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Structured-uncertainty surrogate using worst-case sampled dynamics.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    worst_level = max(float(level) for level in uncertainty_levels)
    a_worst = a * (1.0 + worst_level)
    b_worst = b * max(0.2, 1.0 - 0.5 * worst_level)
    return lqr(a_worst, b_worst, q, r)


def sliding_mode_control(
    state: np.ndarray,
    reference: np.ndarray,
    lambda_gain: float = 1.0,
    reaching_gain: float = 0.6,
    boundary: float = 0.1,
) -> float:
    state = np.asarray(state, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    error = state - reference
    surface = error[1] + lambda_gain * error[0] if error.size >= 2 else error[0]
    sat = np.clip(surface / max(boundary, 1e-9), -1.0, 1.0)
    return -lambda_gain * error[0] - reaching_gain * sat


def backstepping_control(
    position: float,
    velocity: float,
    reference: float,
    k1: float = 1.2,
    k2: float = 1.0,
) -> float:
    z1 = position - reference
    virtual_velocity = -k1 * z1
    z2 = velocity - virtual_velocity
    return -k2 * z2 - z1


def simulate_state_feedback(
    a: np.ndarray,
    b: np.ndarray,
    k: np.ndarray,
    x0: np.ndarray,
    steps: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    xs = [x[:, 0].copy()]
    us = []
    for _ in range(steps):
        u = -k @ x
        x = a @ x + b @ u
        xs.append(x[:, 0].copy())
        us.append(u[:, 0].copy())
    return np.asarray(xs), np.asarray(us)


def quadratic_cost(xs: np.ndarray, us: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    total = 0.0
    for idx in range(len(us)):
        x = xs[idx].reshape(-1, 1)
        u = us[idx].reshape(-1, 1)
        total += (x.T @ q @ x + u.T @ r @ u).item()
    x_terminal = xs[-1].reshape(-1, 1)
    total += (x_terminal.T @ q @ x_terminal).item()
    return total


def lyapunov_decay(
    closed_loop: np.ndarray,
    p: np.ndarray,
    samples: Iterable[np.ndarray],
) -> np.ndarray:
    closed_loop = np.asarray(closed_loop, dtype=float)
    p = np.asarray(p, dtype=float)
    derivatives = []
    for sample in samples:
        x = np.asarray(sample, dtype=float).reshape(-1, 1)
        x_next = closed_loop @ x
        v_now = (x.T @ p @ x).item()
        v_next = (x_next.T @ p @ x_next).item()
        derivatives.append(v_next - v_now)
    return np.asarray(derivatives)


def region_of_attraction_estimate(
    closed_loop: np.ndarray,
    p: np.ndarray,
    radius_samples: Iterable[float],
    angles: int = 16,
) -> float:
    """
    Return the largest sampled radius for which quadratic Lyapunov decay holds
    on a 2D ring sweep.
    """

    closed_loop = np.asarray(closed_loop, dtype=float)
    p = np.asarray(p, dtype=float)
    best_radius = 0.0
    for radius in radius_samples:
        samples = []
        for idx in range(angles):
            theta = 2.0 * np.pi * idx / angles
            samples.append(np.array([radius * np.cos(theta), radius * np.sin(theta)]))
        decay = lyapunov_decay(closed_loop, p, samples)
        if np.all(decay < 0.0):
            best_radius = float(radius)
        else:
            break
    return best_radius


def stability_report(
    controller_name: str,
    closed_loop: np.ndarray,
    p: np.ndarray,
    radius_samples: Iterable[float],
) -> dict[str, float | str]:
    roa = region_of_attraction_estimate(closed_loop, p, radius_samples)
    return {
        "controller": controller_name,
        "region_of_attraction": roa,
        "lyapunov_trace": float(np.trace(p)),
    }
