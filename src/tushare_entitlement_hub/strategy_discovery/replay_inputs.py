"""Declared, portable Return80 replay data; never execute code from a data directory."""

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..regime_research.accounting import Distribution, load_distributions
from ..regime_research.attribution import factor_series
from ..regime_research.market import classify_regimes
from ..regime_research.panel import load_panel
from .breadth_candidate import breadth_low_score_candidate


def file_digest(path: Path) -> str:
    """Stream a file hash; an empty file returns the standard SHA256 empty digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sector_capped_mask(
    eligible: NDArray[np.bool_],
    scores: NDArray[Any],
    sectors: NDArray[Any],
    cap: int,
    *,
    count: int = 10,
) -> NDArray[np.bool_]:
    """Rank each date with stable ties and industry slots before common qualification.

    Example: scores [9,8,7], sectors [1,1,2], cap=1/count=2 select [True,False,True].
    Unknown sectors share one bucket; no future data or holdings enter this selection.
    """
    if (
        eligible.ndim != 2
        or eligible.dtype != bool
        or eligible.shape != scores.shape
        or sectors.shape != scores.shape
        or type(cap) is not int
        or cap not in (1, 2, 3)
        or type(count) is not int
        or count < 1
    ):
        raise ValueError("invalid_sector_cap_inputs")
    selected = np.zeros(eligible.shape, dtype=bool)
    for day in range(len(eligible)):
        columns = np.flatnonzero(eligible[day] & np.isfinite(scores[day]))
        ranked = columns[np.argsort(-scores[day, columns], kind="stable")]
        counts: Counter[int] = Counter()
        total = 0
        for column in ranked:
            value = sectors[day, column]
            sector = int(value) if np.isfinite(value) and value >= 0 else -1
            if counts[sector] >= cap:
                continue
            selected[day, column] = True
            counts[sector] += 1
            total += 1
            if total == count:
                break
    return selected


def verified_replay_manifest(root: Path) -> dict[str, Any]:
    """Verify relative files before loading; a modified NPY or escaping path fails closed.

    Example: a dataset with replay.json and matching SHA entries returns its manifest.
    No Python source, pickle, credentials or absolute data paths are interpreted.
    """
    manifest: dict[str, Any] = json.loads((root / "replay.json").read_text())
    if manifest.get("schema") != "return80-replay-inputs/v1" or not manifest.get("files"):
        raise ValueError("invalid_replay_manifest")
    required = {
        "panel/panel.json",
        "eligible.npy",
        "technical_risk.npy",
        "legacy_scores.npy",
        "distributions.json",
    }
    if not required <= manifest["files"].keys():
        raise ValueError("incomplete_replay_manifest")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("replay_path_outside_dataset")
        if file_digest(path) != expected:
            raise ValueError("changed_replay_input")
    panel = json.loads((root / "panel/panel.json").read_text())
    for group in ("values", "signals", "market"):
        for name in panel[group]:
            if f"panel/{group}-{name}.npy" not in manifest["files"]:
                raise ValueError("unhashed_panel_array")
    return manifest


def load_replay_context(root: Path) -> tuple[Any, ...]:
    """Load a portable manifest or the original cache layout using installed source only.

    Example: root/replay.json relocates to any machine with identical data hashes.
    Legacy cache support reads the original NPY/JSON files, never cache/run.py.
    Returns panel, breadth states, account policy, strategy, distributions, stages, factors.
    """
    portable = (root / "replay.json").exists()
    if portable:
        verified_replay_manifest(root)
        panel_root = root / "panel"
        eligible_path, technical_path, score_path = (
            root / name for name in ("eligible.npy", "technical_risk.npy", "legacy_scores.npy")
        )
    else:
        panel_root = root / "p03-regime-alpha-20261018/panel"
        eligible_path = root / "stockops-pit-redesign-20261018/eligible.npy"
        technical_path = root / "stockops-pit-validation-20261018/matrices/technical_risk.npy"
        score_path = root / "stockops-pit-redesign-20261018/scores.npy"
    panel = load_panel(panel_root)
    eligible = np.load(eligible_path, mmap_mode="r", allow_pickle=False)
    technical = np.load(technical_path, mmap_mode="r", allow_pickle=False)
    scores = np.load(score_path, mmap_mode="r", allow_pickle=False).astype(float)
    shape = (len(panel.calendar), len(panel.codes))
    if eligible.dtype != bool or any(
        values.shape != shape for values in (eligible, technical, scores)
    ):
        raise ValueError("invalid_replay_axes")
    base, states, policy = breadth_low_score_candidate(
        eligible,
        technical == 0,
        scores,
        panel.market["breadth"],
    )
    if portable:
        events = [
            Distribution(**row) for row in json.loads((root / "distributions.json").read_text())
        ]
    else:
        events = load_distributions(
            root / "p03-regime-alpha-20261018/raw", panel.calendar, panel.codes
        )
    original_states = classify_regimes(
        panel.market["index"],
        panel.market["breadth"],
        panel.market["advancing"],
        panel.market["amount"],
        panel.market["limit_down"],
    )
    return (
        panel,
        states,
        policy,
        replace(base, entry_mask=eligible & (technical == 0)),
        events,
        original_states,
        factor_series(panel),
    )


def export_replay_inputs(cache: Path, output: Path) -> Path:
    """Copy the original replay data closure and hash it; refuse an existing destination.

    Example: export_replay_inputs(cache, fresh) writes fresh/replay.json and its arrays.
    Parsed corporate actions retain receivable/listing indices beyond the final session.
    Frozen predictions are supplied separately by the chosen model, not embedded here.
    """
    if output.exists():
        raise ValueError("fresh_replay_output_required")
    context = load_replay_context(cache)
    events = context[4]
    source_panel = cache / "p03-regime-alpha-20261018/panel"
    files = {
        "panel/panel.json": source_panel / "panel.json",
        "eligible.npy": cache / "stockops-pit-redesign-20261018/eligible.npy",
        "technical_risk.npy": cache
        / "stockops-pit-validation-20261018/matrices/technical_risk.npy",
        "legacy_scores.npy": cache / "stockops-pit-redesign-20261018/scores.npy",
    }
    return write_replay_inputs(source_panel, files, events, output)


def write_replay_inputs(
    source_panel: Path,
    input_files: dict[str, Path],
    events: list[Distribution],
    output: Path,
) -> Path:
    """Freeze a declared panel/candidate/corporate-action closure without a legacy layout.

    Example: rebuilt eligible/technical/legacy-score paths produce a relocatable replay
    directory. All panel arrays are copied and hashed, then the loader verifies the closure.
    """
    if output.exists():
        raise ValueError("fresh_replay_output_required")
    panel = load_panel(source_panel)
    files = dict(input_files)
    if set(files) != {
        "panel/panel.json",
        "eligible.npy",
        "technical_risk.npy",
        "legacy_scores.npy",
    }:
        raise ValueError("invalid_replay_file_roles")
    metadata = json.loads((source_panel / "panel.json").read_text())
    for group in ("values", "signals", "market"):
        for name in metadata[group]:
            filename = f"{group}-{name}.npy"
            files[f"panel/{filename}"] = source_panel / filename
    output.mkdir(parents=True)
    for relative, source in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (output / "distributions.json").write_text(json.dumps([asdict(event) for event in events]))
    manifest = {
        "schema": "return80-replay-inputs/v1",
        "historical_pit_certified": False,
        "calendar": [panel.calendar[0], panel.calendar[-1]],
        "securities": len(panel.codes),
        "files": {
            relative: file_digest(output / relative) for relative in (*files, "distributions.json")
        },
    }
    (output / "replay.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verified_replay_manifest(output)
    return output / "replay.json"
