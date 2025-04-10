# traffick_fluo/pipeline/cli.py

import argparse

from traffick_fluo.pipeline.runner import run_segment, run_train, run_score, run_prepare
from traffick_fluo.utils.logging import setup_logger


def main() -> None:
    """
    Entry point for the Cell Image Classification Pipeline CLI.

    Supports subcommands:
        - prepare : Run per-stage input prep (membrane/transfection only)
        - segment : Run segmentation and feature extraction
        - train   : Train a model for a given stage
        - score   : Score features using a trained model
        - rescore : Retrain and score using same features

    All commands require a YAML config file.
    """
    parser = argparse.ArgumentParser(description="Cell Image Pipeline CLI")

    # Global args
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Prepare inputs for membrane or transfection stage")
    subparsers.add_parser("segment", help="Run segmentation and extract features")
    subparsers.add_parser("train", help="Train model for a given stage")
    subparsers.add_parser("score", help="Score features using the latest trained model")
    subparsers.add_parser("rescore", help="Retrain and score using existing features")

    args = parser.parse_args()

    # Logging
    setup_logger(level="DEBUG" if args.verbose else "INFO")

    if not args.config:
        parser.print_help()
        raise ValueError("--config is required for all commands")

    # Dispatch by command
    if args.command == "prepare":
        run_prepare(args.config)
    elif args.command == "segment":
        run_segment(args.config)
    elif args.command == "train":
        run_train(config_path=args.config)
    elif args.command == "score":
        run_score(args.config)
    elif args.command == "rescore":
        run_train(config_path=args.config, rescore=True)
