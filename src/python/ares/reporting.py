from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_results(base_dir: str | Path = "results") -> dict[str, object]:
    base_dir = Path(base_dir)

    assembly_rows = _read_csv(base_dir / "assembly" / "latency_comparison.csv")
    nav_success_rows = _read_csv(base_dir / "ares_navigator" / "navigation_success_rate.csv")
    nav_return_rows = _read_csv(base_dir / "ares_navigator" / "mpc_vs_rl_comparison.csv")
    planning_rows = _read_csv(base_dir / "planning" / "algorithm_comparison.csv")
    motion_rows = _read_csv(base_dir / "planning" / "motion_primitives.csv")
    control_rows = _read_csv(base_dir / "control" / "stability_analysis.csv")
    go_rows = _read_csv(base_dir / "go" / "fleet_benchmark.csv")
    simulation_rows = _read_csv(base_dir / "simulation" / "scenario_results.csv")
    causal_rows = _read_csv(base_dir / "ares_navigator" / "causal_failure_analysis.csv")

    fastest_impl = min(assembly_rows, key=lambda row: float(row["latency_ns"]))
    best_policy = max(nav_success_rows, key=lambda row: float(row["success_rate"]))
    best_return_policy = max(nav_return_rows, key=lambda row: float(row["mean_return"]))
    best_motion = min(motion_rows, key=lambda row: float(row["path_length"]))
    stable_rows = [row for row in control_rows if row["stable"].lower() == "true"]
    unstable_rows = [row for row in control_rows if row["stable"].lower() != "true"]
    largest_fleet = max(go_rows, key=lambda row: int(row["fleet_size"]))
    successful_scenarios = [row["scenario"] for row in simulation_rows if float(row["success_rate"]) >= 1.0]

    planning_by_size: dict[str, dict[str, str]] = {}
    for row in planning_rows:
        planning_by_size.setdefault(row["map_size"], row)
        if float(row["path_length"]) < float(planning_by_size[row["map_size"]]["path_length"]):
            planning_by_size[row["map_size"]] = row

    summary = {
        "assembly": {
            "fastest_implementation": fastest_impl["implementation"],
            "latency_ns": float(fastest_impl["latency_ns"]),
        },
        "navigator": {
            "best_success_policy": best_policy["policy"],
            "best_success_rate": float(best_policy["success_rate"]),
            "best_return_policy": best_return_policy["policy"],
            "best_return": float(best_return_policy["mean_return"]),
            "causal_failure_total": int(sum(int(row["count"]) for row in causal_rows)),
        },
        "planning": {
            "best_by_map_size": {
                size: {
                    "algorithm": row["algorithm"],
                    "path_length": float(row["path_length"]),
                    "computation_time_ms": float(row["computation_time_ms"]),
                }
                for size, row in sorted(planning_by_size.items())
            },
            "best_motion_primitive": {
                "primitive": best_motion["primitive"],
                "path_length": float(best_motion["path_length"]),
            },
        },
        "control": {
            "stable_samples": len(stable_rows),
            "unstable_samples": len(unstable_rows),
        },
        "fleet": {
            "largest_fleet_size": int(largest_fleet["fleet_size"]),
            "task_completion_s": float(largest_fleet["task_completion_s"]),
            "communication_overhead_ms": float(largest_fleet["communication_overhead_ms"]),
        },
        "simulation": {
            "successful_scenarios": successful_scenarios,
            "scenario_count": len(simulation_rows),
        },
    }
    return summary


def write_results_summary(base_dir: str | Path = "results") -> dict[str, object]:
    base_dir = Path(base_dir)
    summary = summarize_results(base_dir)
    (base_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = [
        "# ARES Results Summary",
        "",
        f"- Fastest control kernel: `{summary['assembly']['fastest_implementation']}` at `{summary['assembly']['latency_ns']:.0f} ns`.",
        f"- Best navigation success: `{summary['navigator']['best_success_policy']}` at `{summary['navigator']['best_success_rate']:.2f}`.",
        f"- Best navigation return: `{summary['navigator']['best_return_policy']}` at `{summary['navigator']['best_return']:.2f}`.",
        f"- Stable control samples: `{summary['control']['stable_samples']}` with `{summary['control']['unstable_samples']}` unstable samples.",
        f"- Largest fleet benchmark: `{summary['fleet']['largest_fleet_size']}` robots with `{summary['fleet']['communication_overhead_ms']:.1f} ms` communication overhead.",
        f"- Successful simulation scenarios: `{', '.join(summary['simulation']['successful_scenarios']) or 'none'}`.",
    ]
    (base_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    write_results_summary()
