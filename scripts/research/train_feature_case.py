"""Fit one predeclared feature case; never choose features or trading rules by results."""

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from tushare_entitlement_hub.strategy_discovery.boosted_tree_scoring import BoostedTreeModel
from tushare_entitlement_hub.strategy_discovery.feature_experiments import (
    portable_numeric_model,
    stratified_feature_placebo,
)
from tushare_entitlement_hub.strategy_discovery.frozen_feature_combination import (
    combine_frozen_feature_blocks,
)
from tushare_entitlement_hub.strategy_discovery.joint_exit_scoring import exit_training_targets
from tushare_entitlement_hub.strategy_discovery.model_capacity import resolve_model_capacity


def digest(path: Path) -> str:
    """Hash an immutable artifact; for example a pinned NPY yields 64 hexadecimal digits."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Fit one named case from a SHA-pinned matrix specification and export verified scores.

    Example: --spec experiment.json --case event+quality creates only that case in
    spec.output_root. Existing output fails closed, preventing silent replacement.
    Every model uses the original full-sample labels and daily weights. Optional
    spec.model_capacity explicitly overrides only tree depth, leaves or rounds;
    omitted overrides preserve the original parameters exactly.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.case not in spec["cases"]:
        raise ValueError("unregistered experiment case")
    root = Path(spec["output_root"]) / args.case
    if root.exists():
        raise ValueError("fresh case output required")
    cache = Path(spec["cache_root"])
    for item in spec["pinned_inputs"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("changed pinned input: " + Path(item["path"]).name)
    if sklearn.__version__ != spec["sklearn_version"]:
        raise ValueError("unfrozen sklearn version")
    base_root = cache / "stockops-return80-model-20261019"
    metadata = json.loads((base_root / "models/hgb_leaves31_rounds1600.json").read_text())
    parameters = resolve_model_capacity(metadata["parameters"], spec.get("model_capacity"))
    base_features = np.load(base_root / "features.npy", mmap_mode="r")
    axis = np.load(cache / "stockops-utility-score-20261019/candidate-axis.npz")
    panel = json.loads((cache / "p03-regime-alpha-20261018/panel/panel.json").read_text())
    calendar = panel["calendar"]
    labels = np.load(cache / "stockops-joint-score-exit-20261019/long-labels.npy", mmap_mode="r")
    closed, targets = exit_training_targets(labels)
    mature = (axis["days"] >= calendar.index("20211018")) & (axis["days"] < len(calendar) - 31)
    fit = closed & mature
    if int(fit.sum()) != spec["fit_rows"]:
        raise ValueError("changed mature fit axis")
    response = targets[mature[closed], 0]
    _, inverse, counts = np.unique(axis["days"][fit], return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse].astype(float)
    weights /= weights.mean()
    items = [spec["baseline_additional"]] + [
        spec["groups"][key] for key in spec["cases"][args.case]
    ]
    blocks = tuple(
        (Path(item["path"]), item["sha256"], tuple(item["feature_names"])) for item in items
    )
    additional, additional_names = combine_frozen_feature_blocks(
        blocks, expected_rows=spec["candidate_rows"]
    )
    placebo_audit = None
    if args.case == "event_placebo":
        sector_path = Path(spec["candidate_sectors"]["path"])
        if digest(sector_path) != spec["candidate_sectors"]["sha256"]:
            raise ValueError("changed placebo sector axis")
        additional[:, 6:], placebo_audit = stratified_feature_placebo(
            additional[:, 6:], axis["days"], np.load(sector_path), seed=20261019
        )
    features = np.column_stack((base_features, additional)).astype(np.float32)
    names = tuple(metadata["feature_names"]) + additional_names
    if len(set(names)) != len(names) or np.isinf(features).any():
        raise ValueError("invalid combined feature contract")
    root.mkdir(parents=True)
    protocol = {
        "case": args.case,
        "groups": spec["cases"][args.case],
        "feature_names": names,
        "spec_sha256": digest(args.spec),
        "source_revision": spec["source_revision"],
        "script_sha256": digest(Path(__file__)),
        "fit_rows": int(fit.sum()),
        "candidate_rows": len(features),
        "parameters": parameters,
        "model_capacity": spec.get("model_capacity"),
        "sklearn_version": sklearn.__version__,
        "blocks": items,
        "placebo": placebo_audit,
        "historical_pit_certified": False,
        "sample": "full five-year fit and selection; no financial holdout",
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    started = time.monotonic()
    print(json.dumps({"fit_start": args.case, "columns": len(names)}), flush=True)
    model = HistGradientBoostingRegressor(**parameters)
    model.fit(features[fit], response, sample_weight=weights)
    prediction = model.predict(features)
    portable = portable_numeric_model(model, names)
    restored = BoostedTreeModel.from_dict(portable)
    verified = restored.predict(features, feature_names=names)
    np.testing.assert_array_equal(prediction, verified)
    model_path = root / "model-portable.json"
    model_path.write_text(json.dumps(portable, separators=(",", ":"), allow_nan=False))
    with (root / "model.pickle").open("wb") as stream:
        pickle.dump(model, stream, protocol=5)
    np.save(root / "candidate-prediction.npy", prediction)
    matrix = np.full((len(calendar), len(panel["codes"])), np.nan)
    matrix[axis["days"], axis["securities"]] = prediction
    score_path = root / "scores.npy"
    np.save(score_path, matrix)
    audit = {
        "case": args.case,
        "seconds": time.monotonic() - started,
        "columns": len(names),
        "trees": len(portable["trees"]),
        "fit_rows": int(fit.sum()),
        "portable_prediction_max_error": 0.0,
        "additional_fit_finite_counts": np.isfinite(additional[fit]).sum(axis=0).tolist(),
        "model_sha256": digest(model_path),
        "score_sha256": digest(score_path),
    }
    (root / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
