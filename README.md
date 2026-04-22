# ARES


ARES is a lightweight autonomous robotics research framework that combines
classical control, sampling-based planning, hybrid MPC and RL navigation,
real-time assembly kernels, and a Go fleet coordination layer.

The current repository is an implementation-focused research baseline with
working cross-language modules, generated result artifacts, and automated tests.


## Project Metadata

| Field | Value |
| --- | --- |
| Author | George David Tsitlauri |
| Affiliation | Dept. of Informatics & Telecommunications, University of Thessaly, Greece |
| Contact | gdtsitlauri@gmail.com |
| Year | 2026 |

## Primary Research Thesis

ARES is strongest as a bounded cross-language robotics baseline that brings
control, planning, hybrid navigation, and lightweight fleet coordination into a
single reproducible repository. The clearest empirical evidence in the current
 release is not a claim of full-stack production autonomy, but a compact set of
 repeatable artifacts showing that:

- the assembly control path is materially faster than the C and Python
  baselines,
- the hybrid ARES-NAVIGATOR stack improves over standalone MPC and RL
  baselines on the committed navigation scenarios,
- the fleet-coordination path scales to the repository's current 50-robot
  benchmark envelope.

## Evidence Snapshot

The committed artifacts in `results/summary.json` currently show:

- Assembly PID latency: `180 ns`
- C PID latency: `520 ns`
- Python PID latency: `3800 ns`
- ARES-NAVIGATOR success rate: `0.93`
- MPC success rate: `0.86`
- RL success rate: `0.81`
- Largest benchmarked fleet size: `50`
- Fleet overhead at 50 robots: `27.5 ms`

The planning suite also records multi-scale behavior across `50x50`, `100x100`,
and `500x500` maps. These artifacts are useful for comparing planner behavior
inside the repository, but they should be read as bounded benchmark evidence
rather than as a universal claim that one planner dominates every map regime.

**Implemented**
- `src/python/ares/control.py`: PID, cascade PID, fractional PID, LQR, observer
  utilities, robust-control surrogates, Lyapunov decay, and region-of-attraction
  estimation
- `src/python/ares/planning.py`: Dijkstra, A*, D* Lite wrapper, RRT, RRT*,
  PRM, motion primitives, CHOMP-style smoothing, and TrajOpt-style refinement
- `src/python/ares/ares_navigator.py`: hybrid MPC plus learned policy routing,
  causal failure summaries, and hybrid Lyapunov certificates
- `src/python/ares/simulation.py`: bicycle, differential-drive, and unicycle
  models with lidar, GPS, IMU, camera, and scenario runners
- `src/assembly/`: NASM PID kernel, interrupt simulator, C bridge, and latency
  benchmark harness
- `src/go/`: fleet assignment, RPC service skeleton, consensus state, and
  protobuf schema
- `src/planning/planning_core.cpp`: C++ planning and collision helpers with
  pybind-ready bindings

**Repository Layout**

- `src/python/ares/`: control, planning, simulation, and ARES-NAVIGATOR logic
- `src/assembly/`: x86-64 NASM PID kernel and C bridge
- `src/planning/`: C++ planning core scaffold
- `src/go/`: fleet coordination and consensus baseline
- `paper/`: IEEE-style manuscript skeleton
- `results/`: generated benchmark and stability artifacts
- `tests/`: Python integration tests for the core stack

**Working Status**
- Safe default verification is green for the Python and Go suites.
- The C++ planning core builds successfully through CMake.
- Heavy artifact regeneration is available, but intentionally separated from the
  default test path to avoid overloading laptop-class WSL2 environments.
- The repository keeps a protobuf schema for fleet coordination, while the
  default Go communication path remains RPC-based because the locally available
  cached gRPC toolchain requires a newer Go version than the pinned workspace
  version.

**Verification**
- Python test suite: `PYTHONPATH=src/python python3 -m unittest discover -s tests -v`
- Go test suite: `GOCACHE=/tmp/ares-go-cache go test ./...`
- Manual artifact regeneration:
  `PYTHONPATH=src/python python3 -m ares.experiments`
- Optional heavy smoke test:
  `ARES_RUN_HEAVY_RESULTS=1 PYTHONPATH=src/python python3 -m unittest tests.test_ares.TestARES.test_result_generation_smoke -v`
- Verified C++ planning-core build:
  `cmake -S src/planning -B build/planning && cmake --build build/planning`

**Artifacts**
- Control: `results/control/*.csv`, `results/control/stability_proofs.txt`
- Planning: `results/planning/algorithm_comparison.csv`,
  `results/planning/motion_primitives.csv`
- Assembly: `results/assembly/latency_comparison.csv`
- Go: `results/go/fleet_benchmark.csv`
- Navigator: `results/ares_navigator/*.csv`,
  `results/ares_navigator/stability_proof.txt`
- Simulation: `results/simulation/scenario_results.csv`

**Auxiliary Tooling**
- `Makefile`: safe targets for tests, results generation, assembly benchmark,
  and C++ build scaffolding
- `src/control/octave/`: Octave helpers for PID response export and state-space
  analysis summaries

**Research Notes**
- The current implementation emphasizes deterministic, bounded experiments over
  long-running training loops.
- Stability support is provided through computational certificates, sampled
  Lyapunov decay, and region-of-attraction estimation rather than full symbolic
  proofs.
- The learned navigation policy is a repository-scale baseline designed to stay
  compatible with rapid local verification.

## Evidence Hierarchy

- Primary evidence: assembly latency, hybrid navigation success, and fleet
  scaling artifacts
- Secondary evidence: control certificates and multi-scale planning comparisons
- Supporting evidence: cross-language architecture, protobuf/RPC integration,
  and deterministic simulation outputs

**Heavy Work Left Open**
- Full Go gRPC deployment is not part of the default path yet because the
  available environment and dependency baseline do not line up cleanly enough
  to keep that stack stable inside the current reproducible workspace.
- A full SAC-style GPU training pipeline was intentionally left out of the
  default implementation because it would require much heavier, longer-running
  experiment budgets than this bounded laptop-scale baseline is designed to run
  safely.
- Strong formal proofs for every controller and for the hybrid switching system
  remain future work because they require a dedicated theoretical treatment
  beyond the computational certificates included in this repository.
- A high-fidelity C++ physics simulator and hardware-grade real-time validation
  campaign remain future work because they would substantially expand the scope
  from a reproducible software baseline into a much larger systems program.

The code favors deterministic, bounded experiments so the repo stays practical
on a laptop-class WSL2 workstation without heavyweight middleware.

Full generated Go gRPC bindings are not enabled in the default build because
the locally cached `grpc/protobuf` toolchain available in this environment
requires a newer Go version than the configured `go1.22.2` workspace. The repo
therefore keeps the `.proto` schema and safe RPC-based coordination path active
by default.


