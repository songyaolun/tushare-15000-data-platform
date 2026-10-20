"""Portable native HGB training with resource inspection and bounded thread selection."""

import json
import math
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..progress import progress_phase
from ..strategy_discovery.replay_inputs import file_digest
from .dataset import load_dataset, map_predictions


def cgroup_directories(
    root: Path,
    membership: Path = Path("/proc/self/cgroup"),
) -> list[Path]:
    """Enumerate visible process cgroups and ancestors; example nested quotas are not skipped."""
    directories = {root, root / "cpu", root / "cpu,cpuacct", root / "memory"}
    if membership.exists():
        for line in membership.read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            bases = [root] if not controllers else [root / controllers, root / "cpu"]
            for base in bases:
                current = base / relative.lstrip("/")
                while current.is_relative_to(root):
                    directories.add(current)
                    if current == root:
                        break
                    current = current.parent
    return sorted(directories)


def cgroup_memory_available(root: Path = Path("/sys/fs/cgroup")) -> list[int]:
    """Return bounded memory remaining at each v1/v2 cgroup; unlimited gives no bound.

    Example: limit=1GiB and current=256MiB leaves 768MiB even on a large host.
    """
    available = []
    for directory in cgroup_directories(root):
        for limit_name, current_name in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            limit, current = directory / limit_name, directory / current_name
            if not limit.is_file() or not current.is_file():
                continue
            try:
                if limit.read_text().strip() != "max":
                    available.append(max(0, int(limit.read_text()) - int(current.read_text())))
            except (OSError, ValueError):
                continue
    return available


def cpu_quotas(root: Path = Path("/sys/fs/cgroup")) -> list[float]:
    """Read visible ancestor v1/v2 CPU quotas; example cpu.max='200000 100000' gives two."""
    quotas = []
    for directory in cgroup_directories(root):
        maximum = directory / "cpu.max"
        quota = directory / "cpu.cfs_quota_us"
        period = directory / "cpu.cfs_period_us"
        try:
            if maximum.is_file():
                limit, interval = maximum.read_text().split()
                if limit != "max":
                    quotas.append(float(limit) / float(interval))
            elif quota.is_file() and period.is_file() and float(quota.read_text()) > 0:
                quotas.append(float(quota.read_text()) / float(period.read_text()))
        except (OSError, ValueError):
            continue
    return quotas


def inspect_resources() -> dict[str, Any]:
    """Record current affinity, quotas, RAM, load and GPU inventory, without credentials.

    Example: a 64-core host restricted to 4 CPUs yields cpu_limit<=4. GPU inventory
    is informational: sklearn HGB uses CPU and does not claim CUDA acceleration.
    """
    import psutil  # type: ignore[import-untyped]
    from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

    process = psutil.Process()
    affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else []
    quotas = cpu_quotas()
    maximum = min([len(affinity) or os.cpu_count() or 1, *quotas])
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        devices = gpu.stdout.strip().splitlines() if gpu.returncode == 0 else []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        devices = []
    memory = psutil.virtual_memory()
    memory_limits = cgroup_memory_available()
    return {
        "affinity": affinity,
        "cgroup_cpu_quotas": quotas,
        "cpu_limit": max(1, math.floor(maximum)),
        "load_average": list(os.getloadavg()),
        "cpu_busy_percent": psutil.cpu_percent(interval=0.1),
        "available_memory_bytes": min([memory.available, *memory_limits]),
        "cgroup_memory_available_bytes": memory_limits,
        "gpu_inventory": devices,
        "algorithm_gpu_supported": False,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "threadpools": threadpool_info(),
    }


def select_threads(
    features: Any,
    targets: Any,
    weights: Any,
    parameters: dict[str, Any],
    resource: dict[str, Any],
    ceiling: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Time equal four-round probes on all actual fit rows and verify predictions.

    Example: if two threads beat one and four, returns (2, timings). All probes use
    the same deterministic input/seed; BLAS is capped at one to prevent nesting.
    A user ceiling is a bound, not an unmeasured performance claim.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
    from threadpoolctl import threadpool_limits

    if ceiling < 1:
        raise ValueError("invalid_thread_ceiling")
    idle = max(1, math.floor(resource["cpu_limit"] * (1 - resource["cpu_busy_percent"] / 100)))
    maximum = min(ceiling, resource["cpu_limit"], idle)
    candidates = sorted({1, maximum, *(value for value in (2, 4, 8, 16) if value <= maximum)})
    rows = np.linspace(0, len(features) - 1, len(features), dtype=int)
    sample, response, sample_weight = features[rows], targets[rows], weights[rows]
    settings = {**parameters, "max_iter": min(4, parameters["max_iter"]), "verbose": 0}
    timings: list[dict[str, Any]] = []
    reference = None
    for threads in candidates:
        started = time.monotonic()
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=threads, user_api="openmp"):
                model = HistGradientBoostingRegressor(**settings)
                model.fit(sample, response, sample_weight=sample_weight)
                prediction = model.predict(sample)
        if reference is None:
            reference = prediction
        else:
            np.testing.assert_allclose(prediction, reference, atol=1e-12, rtol=1e-12)
        result = {"threads": threads, "seconds": time.monotonic() - started, "rows": len(rows)}
        timings.append(result)
        print(json.dumps({"stage": "thread_probe", **result}), flush=True)
    return min(timings, key=lambda item: item["seconds"])["threads"], timings


def train(dataset: Path, output: Path, *, rounds: int, thread_ceiling: int = 16) -> Path:
    """Fit all mature rows with frozen half126 weights and save native model/scores first.

    Example: train(dataset, run, rounds=4) is a smoke run; rounds=51200 explicitly
    requests the original capacity. Extra aligned feature columns are accepted. No
    financial holdout, portable export, trading, provider call or automatic replay occurs.
    """
    import sklearn  # type: ignore[import-untyped]
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    if rounds < 1 or thread_ceiling < 1 or output.exists():
        raise ValueError("invalid_training_request_or_existing_output")
    with progress_phase("load_training_inputs"):
        arrays, manifest = load_dataset(dataset)
    resources = inspect_resources()
    print(json.dumps({"stage": "resources", **resources}), flush=True)
    estimate = arrays["X"].nbytes * 6 + 256 * 1024**2
    if resources["available_memory_bytes"] < estimate:
        raise ValueError("insufficient_training_memory")
    parameters = {
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": rounds,
        "max_leaf_nodes": 31,
        "max_depth": 5,
        "min_samples_leaf": 1000,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 20261019,
        "max_bins": 255,
        "verbose": 1,
    }
    features = arrays["X"][arrays["fit_indices"]]
    output.mkdir(parents=True)
    with progress_phase("thread_selection"):
        threads, timings = select_threads(
            features, arrays["y"], arrays["sample_weight"], parameters, resources, thread_ceiling
        )
    protocol = {
        "schema": "return80-training-run/v1",
        "dataset_sha256": file_digest(dataset / "dataset.json"),
        "feature_names": manifest["feature_names"],
        "fit_rows": len(features),
        "candidate_rows": len(arrays["X"]),
        "parameters": parameters,
        "half_life_sessions": manifest["half_life_sessions"],
        "financial_holdout": False,
        "historical_pit_certified": False,
        "resources": resources,
        "thread_probes": timings,
        "selected_threads": threads,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "replay_score_source": "native_predictions",
        "portable_export": "not_requested",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with threadpool_limits(limits=1, user_api="blas"):
        with threadpool_limits(limits=threads, user_api="openmp"):
            model = HistGradientBoostingRegressor(**parameters)
            with progress_phase("fit", total_rounds=rounds):
                model.fit(features, arrays["y"], sample_weight=arrays["sample_weight"])
            with progress_phase("native_checkpoint"):
                with (output / "model.pickle").open("wb") as stream:
                    pickle.dump(model, stream, protocol=5)
            with progress_phase("native_predict", total_rows=len(arrays["X"])):
                prediction = model.predict(arrays["X"])
                np.save(output / "candidate-prediction.npy", prediction, allow_pickle=False)
    with progress_phase("map_predictions"):
        map_predictions(dataset, output / "candidate-prediction.npy", output / "scores.npy")
    result = output / "complete.json"
    result.write_text(
        json.dumps(
            {
                "status": "native_ready",
                "files": {
                    name: file_digest(output / name)
                    for name in (
                        "protocol.json",
                        "model.pickle",
                        "candidate-prediction.npy",
                        "scores.npy",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    return result
