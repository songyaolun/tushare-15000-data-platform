"""Installed entry points for portable research; live collection is always explicit."""

import argparse
import json
from pathlib import Path

from ..client import Client, HttpTransport
from ..credentials import configured_credential
from ..raw import RawStore
from ..strategy_discovery.replay_inputs import export_replay_inputs
from .collection import collect_plan, load_collection_plan
from .dataset import append_features, import_training_bundle, map_predictions
from .features import build_daily_block


def main(argv: list[str] | None = None) -> None:
    """Dispatch explicit offline stages or an opted-in collection request.

    Example: return80-research import-training --bundle extracted --output dataset.
    replay forwards to the original economic engine; no production notification or trade sink.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("import-training")
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("append-features")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--block", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("map-predictions")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("export-replay")
    command.add_argument("--cache-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("build-daily-features")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--collection", type=Path, required=True)
    command.add_argument("--recipe", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("collect")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--credential-file", type=Path)
    command.add_argument("--rate-per-minute", type=float, default=60.0)
    command.add_argument("--live", action="store_true")
    command = commands.add_parser("make-plan")
    command.add_argument("--start", required=True)
    command.add_argument("--end", required=True)
    command.add_argument("--bootstrap", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("materialize-sources")
    command.add_argument("--bootstrap", type=Path, required=True)
    command.add_argument("--collection", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("prepare-sources")
    command.add_argument("--sources", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("rebuild-matrices")
    command.add_argument("--spec", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("train")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--rounds", type=int, required=True)
    command.add_argument("--thread-ceiling", type=int, default=16)
    commands.add_parser("replay", add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command == "replay":
        from .replay import main as replay_main

        replay_main(remaining)
        return
    if remaining:
        parser.error("unrecognized arguments")
    if args.command == "import-training":
        result = import_training_bundle(args.bundle, args.output)
    elif args.command == "append-features":
        result = append_features(args.dataset, args.block, args.output)
    elif args.command == "map-predictions":
        result = map_predictions(args.dataset, args.predictions, args.output)
    elif args.command == "export-replay":
        result = export_replay_inputs(args.cache_root, args.output)
    elif args.command == "build-daily-features":
        result = build_daily_block(args.dataset, args.collection, args.recipe, args.output)
    elif args.command == "make-plan":
        from .sources import make_plan

        result = make_plan(args.output, start=args.start, end=args.end, bootstrap=args.bootstrap)
    elif args.command == "materialize-sources":
        from .sources import materialize_sources

        result = materialize_sources(args.bootstrap, args.collection, args.output)
    elif args.command == "prepare-sources":
        from .sources import prepare_sources

        result = prepare_sources(args.sources, args.output)
    elif args.command == "rebuild-matrices":
        from .rebuild import rebuild_matrices

        result = rebuild_matrices(args.spec, args.output)
    elif args.command == "train":
        from .training import train

        result = train(
            args.dataset, args.output, rounds=args.rounds, thread_ceiling=args.thread_ceiling
        )
    else:
        load_collection_plan(args.plan)
        if not args.live:
            raise ValueError("live_collection_requires_explicit_flag")
        with HttpTransport(
            lambda: configured_credential(args.credential_file) or "", live=True
        ) as transport:
            client = Client(
                transport,
                RawStore(args.output / "raw"),
                account="return80-research",
                rate_per_minute=args.rate_per_minute,
            )
            result = collect_plan(client, args.plan, args.output)
    print(json.dumps({"status": "complete", "output": str(result)}), flush=True)
