from __future__ import annotations

import csv
import ctypes
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def python_pid(error: float, integral: float, prev_error: float, kp: float, ki: float, kd: float) -> float:
    return kp * error + ki * integral + kd * (error - prev_error)


def build_shared_library(root: Path) -> Path:
    build_dir = Path(tempfile.mkdtemp(prefix="ares-assembly-"))
    obj = build_dir / "pid_loop.o"
    lib = build_dir / "libares_pid.so"
    subprocess.run(["nasm", "-f", "elf64", str(root / "pid_loop.asm"), "-o", str(obj)], check=True)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", str(root / "assembly_bridge.c"), str(obj), "-o", str(lib)],
        check=True,
    )
    return lib


def benchmark_calls(fn, iterations: int = 20000) -> float:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn(0.2, 0.1, 0.05, 1.2, 0.3, 0.04)
        samples.append(time.perf_counter_ns() - start)
    return statistics.mean(samples)


def main() -> None:
    root = Path(__file__).resolve().parent
    lib_path = build_shared_library(root)
    handle = ctypes.CDLL(str(lib_path))
    handle.assembly_pid.restype = ctypes.c_double
    handle.assembly_pid.argtypes = [ctypes.c_double] * 6
    handle.c_pid.restype = ctypes.c_double
    handle.c_pid.argtypes = [ctypes.c_double] * 6

    rows = [
        {"implementation": "python_pid", "latency_ns": round(benchmark_calls(python_pid), 2)},
        {"implementation": "c_pid", "latency_ns": round(benchmark_calls(handle.c_pid), 2)},
        {"implementation": "assembly_pid", "latency_ns": round(benchmark_calls(handle.assembly_pid), 2)},
    ]

    output = root.parents[1] / "results" / "assembly" / "latency_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle_out:
        writer = csv.DictWriter(handle_out, fieldnames=["implementation", "latency_ns"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
