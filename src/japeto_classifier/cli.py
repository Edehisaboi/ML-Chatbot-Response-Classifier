from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import uvicorn

from japeto_classifier.config import get_settings
from japeto_classifier.data import ingest_dataset, processed_records_are_current, workbook_schema
from japeto_classifier.embeddings import embedding_artifact_is_current, generate_embeddings
from japeto_classifier.splitting import write_split_assignments
from japeto_classifier.training import evaluation_summary, train_catalogue


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    source = Path(args.source).resolve() if args.source else settings.raw_dataset
    if args.schema_only:
        _print(workbook_schema(source))
        return
    _print(ingest_dataset(source, settings.records_path))


def _split(_: argparse.Namespace) -> None:
    settings = get_settings()
    if not settings.records_path.exists():
        raise SystemExit("Processed records are missing. Run `python -m japeto_classifier ingest` first.")
    _print(write_split_assignments(settings.records_path, settings.splits_path, settings.random_state))


def _embed(args: argparse.Namespace) -> None:
    settings = get_settings()
    if not settings.records_path.exists():
        raise SystemExit("Processed records are missing. Run ingestion first.")
    modes = ["response_only", "context_enhanced"] if args.mode == "all" else [args.mode]
    results = [
        generate_embeddings(
            settings.records_path,
            settings.embeddings_dir,
            mode,
            settings.embedding_model,
            settings.embedding_dimensions,
            batch_size=args.batch_size,
            force=args.force,
        )
        for mode in modes
    ]
    _print(results)


def _train(args: argparse.Namespace) -> None:
    settings = get_settings()
    missing = [path for path in [settings.records_path, settings.splits_path] if not path.exists()]
    if missing:
        raise SystemExit("Processed records or split assignments are missing. Run ingest and split first.")
    results = train_catalogue(settings, features=args.features, quick=args.quick, only=args.only)
    _print({"trained": len(results), "models": [item["model_id"] for item in results]})


def _evaluate(_: argparse.Namespace) -> None:
    _print(evaluation_summary(get_settings()))


def _serve(args: argparse.Namespace) -> None:
    uvicorn.run(
        "japeto_classifier.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def _bootstrap(args: argparse.Namespace) -> None:
    settings = get_settings()
    records_refreshed = not processed_records_are_current(
        settings.raw_dataset, settings.records_path
    )
    if records_refreshed:
        _print(ingest_dataset(settings.raw_dataset, settings.records_path))
    if records_refreshed or not settings.splits_path.exists():
        _print(write_split_assignments(settings.records_path, settings.splits_path, settings.random_state))
    embedding_files_exist = all(
        embedding_artifact_is_current(
            settings.records_path,
            settings.embeddings_dir,
            mode,
            settings.embedding_model,
            settings.embedding_dimensions,
        )
        for mode in ["response_only", "context_enhanced"]
    )
    features = "all" if embedding_files_exist else "tfidf"
    if args.with_openai:
        for mode in ["response_only", "context_enhanced"]:
            generate_embeddings(
                settings.records_path,
                settings.embeddings_dir,
                mode,
                settings.embedding_model,
                settings.embedding_dimensions,
                batch_size=args.batch_size,
            )
        features = "all"
    train_catalogue(settings, features=features, quick=args.quick)
    if not args.no_serve:
        uvicorn.run("japeto_classifier.api:app", host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="japeto-classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Validate and normalize the source workbook")
    ingest.add_argument("--source")
    ingest.add_argument("--schema-only", action="store_true")
    ingest.set_defaults(handler=_ingest)

    split = subparsers.add_parser("split", help="Create shared group-aware partitions")
    split.set_defaults(handler=_split)

    embed = subparsers.add_parser("embed", help="Generate resumable OpenAI embedding matrices")
    embed.add_argument("--mode", choices=["all", "response_only", "context_enhanced"], default="all")
    embed.add_argument("--batch-size", type=int, default=64)
    embed.add_argument("--force", action="store_true")
    embed.set_defaults(handler=_embed)

    train = subparsers.add_parser("train", help="Tune, calibrate, evaluate, and register models")
    train.add_argument("--features", choices=["all", "tfidf", "openai"], default="all")
    train.add_argument("--quick", action="store_true", help="Use one parameter choice per model")
    train.add_argument("--only", nargs="*")
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="Print the model registry summary")
    evaluate.set_defaults(handler=_evaluate)

    serve = subparsers.add_parser("serve", help="Start the local API and dashboard")
    serve.add_argument("--host", default="localhost")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=_serve)

    bootstrap = subparsers.add_parser("bootstrap", help="Prepare missing artifacts, train, then serve")
    bootstrap.add_argument("--with-openai", action="store_true")
    bootstrap.add_argument("--quick", action="store_true")
    bootstrap.add_argument("--batch-size", type=int, default=64)
    bootstrap.add_argument("--no-serve", action="store_true")
    bootstrap.add_argument("--host", default="localhost")
    bootstrap.add_argument("--port", type=int, default=8000)
    bootstrap.set_defaults(handler=_bootstrap)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
