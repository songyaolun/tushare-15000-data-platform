"""Run a SHA-pinned, bounded native-model diagnostic without training or sinks."""

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from tushare_entitlement_hub.strategy_discovery.feature_experiments import (
    stratified_feature_placebo,
)
from tushare_entitlement_hub.strategy_discovery.feature_importance import (
    daily_diagnostics,
    tree_usage,
)


def digest(path: Path) -> str:
    """Hash an immutable input incrementally; input a file, output its SHA256."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    """Analyze one frozen protocol; e.g. --protocol spec.json creates result JSONL.

    All pinned bytes are checked before pickle loading. Prediction baselines must
    match frozen native scores exactly. Each completed perturbation is flushed to
    disk and announced; a failure leaves its completed prefix for inspection.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for item in protocol["pinned_inputs"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("changed pinned input: " + Path(item["path"]).name)
    output = Path(protocol["output"])
    output.mkdir()
    axis = np.load(protocol["axis"])
    calendar = json.loads(Path(protocol["panel"]).read_text())["calendar"]
    selected = np.isin(axis["days"], [calendar.index(day) for day in protocol["dates"]])
    features = np.column_stack([
        np.load(path, mmap_mode="r")[selected] for path in protocol["feature_blocks"]
    ]).astype(np.float32)
    days = axis["days"][selected]
    sectors = np.load(protocol["sectors"], mmap_mode="r")[selected]
    labels = np.load(protocol["labels"], mmap_mode="r")[selected]
    mature = ((days <= calendar.index(protocol["signal_cutoff"])) & (labels[:, 3] == 3)
              & (labels[:, 2] <= calendar.index(protocol["exit_cutoff"])))
    common = axis["common"][selected].astype(bool)
    if features.shape != (protocol["rows"], len(protocol["feature_names"])):
        raise ValueError("changed bounded feature shape")
    if np.isinf(features).any() or not np.isfinite(labels[mature, 0]).all():
        raise ValueError("invalid feature or mature label values")
    entries = [("feature", name, [index])
               for index, name in enumerate(protocol["feature_names"])]
    entries += [("group", name, columns) for name, columns in protocol["groups"].items()]
    expected = {(name, kind, feature_name, seed): columns
                for name in protocol["models"] for kind, feature_name, columns in entries
                for seed in protocol["seeds"]}
    resumed = {}
    for path in protocol.get("resume_records", []):
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            key = (record["model"], record["kind"], record["name"], record["seed"])
            if (key in resumed or key not in expected or record["columns"] != expected[key]
                    or record["permutation"]["rows"] != len(features)):
                raise ValueError("invalid or duplicate resumed record")
            resumed[key] = record
    total = len(protocol["models"]) * len(entries) * len(protocol["seeds"])
    completed = 0
    started = time.monotonic()
    np.savez(output / "sample-axis.npz", days=days, securities=axis["securities"][selected],
             common=common, mature=mature, original_rows=np.flatnonzero(selected))
    for name, paths in protocol["models"].items():
        with Path(paths["model"]).open("rb") as stream:
            model = pickle.load(stream)
        usage = tree_usage(model, protocol["feature_names"])
        (output / f"{name}-usage.json").write_text(json.dumps(usage, indent=2))
        with threadpool_limits(limits=protocol["threads"], user_api="openmp"):
            baseline = model.predict(features)
            frozen = np.load(paths["predictions"], mmap_mode="r")[selected]
            if not np.array_equal(baseline, frozen):
                raise ValueError("native baseline differs from frozen predictions")
            np.save(output / f"{name}-baseline.npy", baseline)
            with (output / f"{name}-permutation.jsonl").open("w") as stream:
                for kind, feature_name, columns in entries:
                    for seed in protocol["seeds"]:
                        key = (name, kind, feature_name, seed)
                        if key in resumed:
                            stream.write(json.dumps(resumed[key], allow_nan=False) + "\n")
                            stream.flush()
                            completed += 1
                            print(json.dumps({"stage": "reused", "model": name,
                                              "completed": completed, "total": total}), flush=True)
                            continue
                        step_start = time.monotonic()
                        changed = features.copy()
                        changed[:, columns], audit = stratified_feature_placebo(
                            features[:, columns], days, np.zeros(len(days), dtype=int), seed=seed,
                        )
                        differences = ~((changed[:, columns] == features[:, columns])
                                        | (np.isnan(changed[:, columns])
                                           & np.isnan(features[:, columns])))
                        audit["value_changed_rows"] = int(np.any(differences, axis=1).sum())
                        prediction = model.predict(changed)
                        record = {"model": name, "kind": kind, "name": feature_name,
                                  "columns": columns, "seed": seed, "permutation": audit,
                                  "days": daily_diagnostics(baseline, prediction, days, sectors,
                                                            common, labels[:, 0], mature),
                                  "seconds": time.monotonic() - step_start}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        completed += 1
                        print(json.dumps({"stage": "permutation", "model": name,
                                          "name": feature_name, "seed": seed,
                                          "completed": completed, "total": total,
                                          "seconds": record["seconds"],
                                          "elapsed": time.monotonic() - started}), flush=True)
    receipt = {"completed": completed, "total": total, "reused": len(resumed),
               "protocol_sha256": digest(args.protocol), "seconds": time.monotonic() - started,
               "artifacts": {path.name: digest(path) for path in sorted(output.iterdir())}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
