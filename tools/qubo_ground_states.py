"""Prove unconstrained Ising ground states with single-threaded SCIP.

Energy is h dot s + sum(J_ij * s_i * s_j for i < j) + offset,
with s in {-1, +1}. The crate's sample_one_real excludes offset.
Binary x uses s = 2*x - 1, with each x_i*x_j linearized.
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pyscipopt import Model, quicksum


def solve(h, edges, j, offset, time_limit=1800.0):
    """Return the original Ising energy and SCIP proof metadata."""
    started = time.monotonic()
    model = Model("ising")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("lp/threads", 1)
    model.setRealParam("limits/time", time_limit)
    model.setRealParam("limits/gap", 0.0)
    model.setRealParam("limits/absgap", 0.0)
    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(len(h))]
    terms = [
        float(value) * (2 * variable - 1) for value, variable in zip(h, x, strict=True)
    ]
    for (u, v), value in zip(edges, j, strict=True):
        product = model.addVar(lb=0.0, ub=1.0, name=f"y_{u}_{v}")
        model.addCons(product <= x[u])
        model.addCons(product <= x[v])
        model.addCons(product >= x[u] + x[v] - 1)
        terms.append(float(value) * (4 * product - 2 * x[u] - 2 * x[v] + 1))
    model.setObjective(quicksum(terms) + offset, "minimize")
    model.optimize()
    energy = None
    if model.getNSols() > 0:
        solution = model.getBestSol()
        spins = [
            1 if model.getSolVal(solution, variable) > 0.5 else -1 for variable in x
        ]
        energy = math.fsum(
            [offset]
            + [float(value) * spin for value, spin in zip(h, spins, strict=True)]
            + [
                float(value) * spins[u] * spins[v]
                for (u, v), value in zip(edges, j, strict=True)
            ]
        )
        assert abs(energy - model.getObjVal()) <= 1e-8 * max(1.0, abs(energy))
    gap = float(model.getGap())
    return {
        "n": len(h),
        "energy": energy,
        "gap": gap if math.isfinite(gap) else None,
        "status": str(model.getStatus()),
        "seconds": time.monotonic() - started,
    }


def fixture_name(source):
    """Keep the family and draw names distinct across both fixture sets."""
    family = source.parents[2].name.removeprefix("portfolio-")
    return f"{family}-{source.parent.name}.qmodel"


def solve_file(source, partial_dir, time_limit):
    """Save each completed solve atomically, so an interrupted batch can resume."""
    filename = fixture_name(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    partial = partial_dir / f"{filename}.json"
    if partial.exists():
        cached = json.loads(partial.read_text())
        if cached["sha256"] == digest and cached["time_limit"] == time_limit:
            return cached["result"]
    with np.load(source, allow_pickle=False) as data:
        result = solve(
            data["h"],
            data["edges"],
            data["j"],
            float(data["offset"].item()),
            time_limit,
        )
    result = {"file": filename, **result}
    write_json(partial, {"sha256": digest, "time_limit": time_limit, "result": result})
    return result


def write_json(path, data):
    """Replace a complete JSON file, without exposing partially written results."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def batch(inputs, output, partial_dir, workers, time_limit):
    """Run one SCIP process per model, with at most 24 simultaneous solves."""
    started = time.monotonic()
    sources = sorted(inputs)
    names = [fixture_name(source) for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("duplicate fixture names")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        futures = [
            pool.submit(solve_file, source, partial_dir, time_limit)
            for source in sources
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{len(results)}/{len(sources)} {result['file']}: "
                f"status={result['status']} gap={result['gap']} seconds={result['seconds']:.3f}",
                flush=True,
            )
    write_json(output, sorted(results, key=lambda result: result["file"]))
    proven = sum(
        result["status"] == "optimal" and result["gap"] == 0 for result in results
    )
    print(
        f"batch: {len(results)} models, {proven} proven, {len(results) - proven} unverified, "
        f"wall_seconds={time.monotonic() - started:.3f}",
        flush=True,
    )


def self_test():
    """Compare SCIP against exhaustive enumeration on three random N=16 models."""
    rng = np.random.default_rng(20261024)
    n = 16
    edges = np.array([(u, v) for u in range(n) for v in range(u + 1, n)])
    spins = 2 * ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) - 1
    for case in range(3):
        # Dyadic coefficients make exact equality meaningful in binary f64.
        h = rng.integers(-16, 17, n) / 8
        j = rng.integers(-16, 17, len(edges)) / 8
        offset = float(rng.integers(-16, 17) / 8)
        energies = spins @ h + offset
        for (u, v), value in zip(edges, j, strict=True):
            energies += value * spins[:, u] * spins[:, v]
        expected = float(energies.min())
        result = solve(h, edges, j, offset)
        assert result["status"] == "optimal" and result["gap"] == 0.0, result
        assert result["energy"] == expected, (case, result, expected)
        print(
            f"self-test {case + 1}/3: N=16 SCIP=brute_force={expected:.17g}", flush=True
        )
    print("self-test: 3 passed", flush=True)


def main():
    """Solve the supplied NPZ files, or cross-check the formulation first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/portfolio/ground_states.json"),
    )
    parser.add_argument("--partial-dir", type=Path, required=False)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.inputs or args.partial_dir is None:
        parser.error("inputs and --partial-dir are required for a batch")
    if (
        not 1 <= args.workers <= 24
        or not math.isfinite(args.time_limit)
        or args.time_limit <= 0
    ):
        parser.error("workers must be 1..24 and time-limit must be positive and finite")
    batch(args.inputs, args.output, args.partial_dir, args.workers, args.time_limit)


if __name__ == "__main__":
    main()
