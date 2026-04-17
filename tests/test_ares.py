from __future__ import annotations

import ctypes
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ares.ares_navigator import HybridNavigator, SimpleMPC, SoftActorCriticLite, hybrid_lyapunov_certificate
from ares.control import (
    CascadePIDController,
    FractionalPIDController,
    FractionalPIDGains,
    PIDController,
    PIDGains,
    backstepping_control,
    kalman_filter_step,
    lqr,
    luenberger_observer_step,
    quadratic_cost,
    simulate_first_order_response,
    simulate_state_feedback,
    sliding_mode_control,
)
from ares.planning import (
    DStarLitePlanner,
    GridMap,
    astar,
    chomp_smooth_path,
    clothoid_curve,
    minimum_snap_trajectory,
    path_length,
    prm,
    reeds_shepp_path,
    rrt,
    rrt_star,
    trajopt_optimize,
)
from ares.simulation import Pose, camera_features, simulate_navigation_episode
from ares.experiments import (
    generate_assembly_results,
    generate_control_results,
    generate_go_results,
    generate_navigator_results,
    generate_planning_results,
    generate_simulation_results,
)
from ares.reporting import summarize_results, write_results_summary


ROOT = Path(__file__).resolve().parents[1]


class TestARES(unittest.TestCase):
    def test_pid_stability(self) -> None:
        controller = PIDController(PIDGains(2.8, 1.2, 0.08))
        trace = simulate_first_order_response(controller, steps=300)
        self.assertLess(abs(trace[-1]["error"]), 0.05)
        self.assertLess(abs(trace[-1]["error"]), abs(trace[0]["error"]))

    def test_lqr_optimal(self) -> None:
        a = np.array([[1.0, 0.1], [0.0, 0.97]])
        b = np.array([[0.0], [0.1]])
        q = np.diag([10.0, 1.0])
        r = np.array([[0.15]])
        k, _ = lqr(a, b, q, r)
        lqr_xs, lqr_us = simulate_state_feedback(a, b, k, np.array([1.0, 0.0]), steps=80)
        lqr_cost = quadratic_cost(lqr_xs, lqr_us, q, r)

        pid = PIDController(PIDGains(1.0, 0.1, 0.05))
        x = np.array([[1.0], [0.0]])
        pid_xs = [x[:, 0].copy()]
        pid_us = []
        for _ in range(80):
            u = pid.step(-float(x[0, 0]), 0.1)
            x = a @ x + b * u
            pid_xs.append(x[:, 0].copy())
            pid_us.append(np.array([u]))
        pid_cost = quadratic_cost(np.asarray(pid_xs), np.asarray(pid_us), q, r)
        self.assertLess(lqr_cost, pid_cost)

    def test_astar_optimal(self) -> None:
        obstacles = frozenset({(1, 1), (1, 2), (2, 1)})
        grid = GridMap(5, 5, obstacles)
        path = astar(grid, (0, 0), (4, 4))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertEqual(path_length(path), 8.0)

    def test_dstar_replans_after_map_update(self) -> None:
        grid = GridMap(6, 6, frozenset())
        planner = DStarLitePlanner(grid)
        initial = planner.plan((0, 0), (5, 5))
        planner.update_obstacles(set(grid.obstacles) | {(0, 1), (0, 2), (1, 1)})
        replanned = planner.plan((0, 0), (5, 5))
        self.assertNotEqual(initial[:3], replanned[:3])
        self.assertEqual(replanned[-1], (5, 5))

    def test_rrt_feasible(self) -> None:
        obstacles = frozenset((3, y) for y in range(8) if y != 4)
        grid = GridMap(8, 8, obstacles)
        path = rrt(grid, (0, 0), (7, 7), iterations=3000, seed=11)
        self.assertIsNotNone(path)
        for node in path or []:
            self.assertTrue(grid.passable(node))

    def test_rrt_star_and_prm_feasible(self) -> None:
        obstacles = frozenset((4, y) for y in range(10) if y != 5)
        grid = GridMap(10, 10, obstacles)
        star_path = rrt_star(grid, (0, 0), (9, 9), iterations=4000, seed=31)
        roadmap_path = prm(grid, (0, 0), (9, 9), samples=80, seed=17)
        self.assertIsNotNone(star_path)
        self.assertEqual(roadmap_path[0], (0, 0))
        self.assertEqual(roadmap_path[-1], (9, 9))

    def test_assembly_pid(self) -> None:
        build_dir = Path(tempfile.mkdtemp(prefix="ares-asm-"))
        asm = ROOT / "src" / "assembly" / "pid_loop.asm"
        bridge = ROOT / "src" / "assembly" / "assembly_bridge.c"
        obj = build_dir / "pid_loop.o"
        lib = build_dir / "libares_pid.so"
        subprocess.run(["nasm", "-f", "elf64", str(asm), "-o", str(obj)], check=True)
        subprocess.run(
            ["gcc", "-shared", "-fPIC", str(bridge), str(obj), "-o", str(lib)],
            check=True,
        )
        handle = ctypes.CDLL(str(lib))
        handle.assembly_pid.restype = ctypes.c_double
        handle.assembly_pid.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        result = handle.assembly_pid(0.5, 0.2, 0.1, 1.2, 0.3, 0.05)
        expected = 1.2 * 0.5 + 0.3 * 0.2 + 0.05 * (0.5 - 0.1)
        self.assertAlmostEqual(result, expected, places=9)

    def test_rl_training(self) -> None:
        trainer = SoftActorCriticLite(device="cpu")
        rewards = trainer.train(episodes=18, batch_size=48)
        self.assertGreater(rewards[-1], rewards[0])

    def test_additional_control_primitives(self) -> None:
        fractional = FractionalPIDController(FractionalPIDGains(1.8, 0.4, 0.05))
        outputs = [fractional.step(1.0 - 0.1 * idx, 0.02) for idx in range(6)]
        self.assertTrue(all(math.isfinite(value) for value in outputs))

        cascade = CascadePIDController(
            outer=PIDController(PIDGains(1.5, 0.2, 0.02)),
            inner=PIDController(PIDGains(2.2, 0.6, 0.05)),
        )
        cascade_out = cascade.step(0.7, 0.1, 0.02)
        self.assertTrue(math.isfinite(cascade_out))

        smc = sliding_mode_control(np.array([0.4, -0.2]), np.zeros(2))
        bstep = backstepping_control(0.3, -0.1, 0.0)
        self.assertTrue(math.isfinite(smc))
        self.assertTrue(math.isfinite(bstep))

    def test_observers_reduce_uncertainty(self) -> None:
        a = np.array([[1.0, 0.1], [0.0, 0.95]])
        b = np.array([[0.0], [0.1]])
        c = np.array([[1.0, 0.0]])
        l = np.array([[0.2], [0.3]])
        x_hat = luenberger_observer_step(a, b, c, l, np.zeros((2, 1)), np.array([[0.1]]), np.array([[1.0]]))
        self.assertEqual(x_hat.shape, (2, 1))

        _, covariance = kalman_filter_step(
            a,
            b,
            c,
            np.diag([1e-3, 1e-3]),
            np.array([[1e-2]]),
            np.zeros((2, 1)),
            np.eye(2),
            np.array([[0.1]]),
            np.array([[1.0]]),
        )
        self.assertLess(np.trace(covariance), 2.0)

    def test_mpc_constraints(self) -> None:
        navigator = HybridNavigator(mpc=SimpleMPC(), rl_policy=SoftActorCriticLite(device="cpu"), confidence_threshold=0.95)
        command = navigator.command(Pose(0.0, 0.0, 0.0), (1.0, 0.0), [(0.3, 0.0)])
        self.assertEqual(command["mode"], "mpc")
        self.assertLessEqual(float(command["velocity"]), navigator.mpc.velocity_limit)
        self.assertLessEqual(abs(float(command["omega"])), navigator.mpc.omega_limit)

    def test_motion_primitives_and_smoothing(self) -> None:
        reeds = reeds_shepp_path((0.0, 0.0), (2.0, 0.0))
        clothoid = clothoid_curve((0.0, 0.0), (2.0, 1.0))
        minimum_snap = minimum_snap_trajectory([(0.0, 0.0), (1.0, 0.7), (2.0, 0.0)])
        smoothed = chomp_smooth_path([(0.0, 0.0), (0.8, 1.2), (1.5, 1.0), (2.0, 0.0)], [(1.0, 0.8)])
        optimized = trajopt_optimize(smoothed, ((0.0, 2.0), (-0.5, 1.5)))
        self.assertEqual(reeds[0], (0.0, 0.0))
        self.assertEqual(reeds[-1], (2.0, 0.0))
        self.assertEqual(clothoid[0], (0.0, 0.0))
        self.assertEqual(minimum_snap[0], (0.0, 0.0))
        self.assertEqual(optimized[0], smoothed[0])
        self.assertEqual(optimized[-1], smoothed[-1])

    def test_simulation_scenario_runs(self) -> None:
        navigator = HybridNavigator(mpc=SimpleMPC(), rl_policy=SoftActorCriticLite(device="cpu"))
        outcome = simulate_navigation_episode(navigator, "maze_navigation", steps=25)
        features = camera_features(Pose(0.0, 0.0, 0.0), [(1.0, 0.0), (3.0, 2.0)])
        self.assertIn("success", outcome)
        self.assertGreaterEqual(len(features), 1)

    def test_lyapunov_stable(self) -> None:
        navigator = HybridNavigator(mpc=SimpleMPC(), rl_policy=SoftActorCriticLite(device="cpu"))
        rows = hybrid_lyapunov_certificate(
            navigator,
            [
                (Pose(0.0, 0.0, 0.0), (1.0, 0.0), []),
                (Pose(0.0, 0.0, 0.0), (1.0, 0.0), [(0.45, 0.0)]),
                (Pose(-0.2, 0.3, -0.1), (0.8, 0.2), [(0.4, 0.4)]),
            ],
        )
        self.assertTrue(all(float(row["dV"]) < 0.0 for row in rows))

    @unittest.skipUnless(
        os.environ.get("ARES_RUN_HEAVY_RESULTS") == "1",
        "Set ARES_RUN_HEAVY_RESULTS=1 to run artifact generation smoke tests.",
    )
    def test_result_generation_smoke(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="ares-results-"))
        generate_control_results(output_dir)
        generate_planning_results(output_dir, sizes=[20, 40])
        generate_assembly_results(output_dir)
        generate_go_results(output_dir)
        generate_navigator_results(output_dir)
        generate_simulation_results(output_dir)
        expected = [
            output_dir / "control" / "pid_response.csv",
            output_dir / "planning" / "algorithm_comparison.csv",
            output_dir / "planning" / "motion_primitives.csv",
            output_dir / "assembly" / "latency_comparison.csv",
            output_dir / "go" / "fleet_benchmark.csv",
            output_dir / "ares_navigator" / "navigation_success_rate.csv",
            output_dir / "simulation" / "scenario_results.csv",
        ]
        for path in expected:
            self.assertTrue(path.exists(), msg=f"missing artifact: {path}")

    def test_results_reporting_summary(self) -> None:
        summary = summarize_results(ROOT / "results")
        self.assertIn("assembly", summary)
        self.assertIn("navigator", summary)
        self.assertIn("planning", summary)
        self.assertEqual(summary["assembly"]["fastest_implementation"], "assembly_pid")

        output_dir = Path(tempfile.mkdtemp(prefix="ares-report-"))
        (output_dir / "assembly").mkdir(parents=True)
        (output_dir / "ares_navigator").mkdir(parents=True)
        (output_dir / "planning").mkdir(parents=True)
        (output_dir / "control").mkdir(parents=True)
        (output_dir / "go").mkdir(parents=True)
        (output_dir / "simulation").mkdir(parents=True)
        generate_control_results(output_dir)
        generate_planning_results(output_dir, sizes=[20])
        generate_assembly_results(output_dir)
        generate_go_results(output_dir)
        generate_navigator_results(output_dir)
        generate_simulation_results(output_dir)
        write_results_summary(output_dir)
        self.assertTrue((output_dir / "summary.json").exists())
        self.assertTrue((output_dir / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
