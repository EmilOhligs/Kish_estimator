#!/usr/bin/env python
"""Entry point for training MACE ensemble members.

Default: train 3 new L2c members (seeds 3, 4, 5) and add them to
models/ensemble_L2c, growing it from 2 to 5 members.

Usage:
    # GPU node:
    python analyses/train_ensemble.py --config configs/ensemble_baseline.yaml
    # Show the mace_run_train commands without running them:
    python analyses/train_ensemble.py --config configs/ensemble_baseline.yaml --dry-run
"""
import argparse

from uq_mace.ensemble import train_ensemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-models", type=int, default=3)
    parser.add_argument("--architecture", default="L2c")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="explicit seeds (default: 3 4 5)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_ensemble(
        args.config,
        n_models=args.n_models,
        seeds=args.seeds,
        architecture=args.architecture,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
