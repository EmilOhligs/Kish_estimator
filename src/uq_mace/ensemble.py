"""MACE ensemble training and inference.

Pre-trained ensembles available under models/ (see models/README.md):
    models/ensemble_L0    3 members, MACE L=0 (invariant)
    models/ensemble_L0c   3 members, MACE L=0, wider channels
    models/ensemble_L2c   2 members, MACE L=2 (equivariant) - matches Hilpert & Kresse 2026
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Hyperparameter der L2c-Architektur (Hilpert & Kresse 2026: L=2, equivariant).
# Werden nur genutzt, wenn die config sie nicht ueberschreibt. Fuer Member, die
# sich NUR im Seed von mace-L2-c-01/02 unterscheiden, hier ggf. Tobis exakte
# Trainings-Hyperparameter eintragen.
L2C_DEFAULTS = dict(
    model="MACE",
    max_L=2,
    correlation=3,
    r_max=6.0,
    num_interactions=2,
    num_channels=128,
    max_ell=3,
    num_radial_basis=8,
    num_cutoff_basis=5,
)


def find_model_paths(model_dir: str | Path) -> list[str]:
    """List all .model checkpoint files in a directory (one ensemble member each)."""
    model_dir = Path(model_dir)
    paths = sorted(str(p) for p in model_dir.glob("*.model"))
    if not paths:
        raise FileNotFoundError(f"No .model files found in {model_dir}")
    return paths


def load_ensemble_calculator(model_dir: str | Path, device: str = "cpu"):
    """Build an ASE calculator from all MACE models in model_dir.

    MACECalculator natively supports multiple model_paths: it evaluates every member and
    exposes both the mean prediction and the per-configuration ensemble spread (used here
    as sigma(R), the ensemble-UQ estimate of Phase 2).
    """
    from mace.calculators import MACECalculator

    model_paths = find_model_paths(model_dir)
    return MACECalculator(model_paths=model_paths, device=device)


def _build_train_command(
    *,
    name: str,
    seed: int,
    train_file: str,
    valid_file: str | None,
    test_file: str | None,
    model_dir: str | Path,
    log_dir: str | Path,
    device: str,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    hparams: dict,
) -> list[str]:
    """Assemble a single `mace_run_train` CLI invocation (one ensemble member)."""
    cmd = [
        "mace_run_train",
        f"--name={name}",
        f"--seed={seed}",
        f"--train_file={train_file}",
        f"--model_dir={model_dir}",
        f"--log_dir={log_dir}",
        f"--checkpoints_dir={log_dir}",
        f"--results_dir={log_dir}",
        f"--device={device}",
        f"--max_num_epochs={max_epochs}",
        f"--batch_size={batch_size}",
        f"--lr={learning_rate}",
    ]
    if valid_file:
        cmd.append(f"--valid_file={valid_file}")
    if test_file:
        cmd.append(f"--test_file={test_file}")
    for key, val in hparams.items():
        cmd.append(f"--{key}={val}")
    return cmd


def train_ensemble(
    config_path: str,
    n_models: int = 3,
    seeds: list[int] | None = None,
    *,
    architecture: str = "L2c",
    device: str = "cuda",
    dry_run: bool = False,
) -> list[str]:
    """Train N new MACE members and add them to an existing ensemble.

    Default: grow ``models/ensemble_L2c`` (currently 2 members) by three new L=2
    equivariant members with seeds 3, 4, 5 -> a 5-member ensemble matching the
    Hilpert & Kresse (2026) architecture. Only the seed varies between members;
    all other hyperparameters come from ``config_path`` (falling back to the L2c
    defaults above). New checkpoints land in ``models/ensemble_L2c`` as
    mace-L2-c-03/04/05.model, next to the existing ones.

    Wraps the MACE CLI ``mace_run_train`` (one process per seed, run sequentially).
    Returns the list of expected output .model paths.

    Parameters
    ----------
    config_path : path to configs/ensemble_baseline.yaml
    n_models : number of new members to train (default 3)
    seeds : explicit seed list; default [3, 4, 5] continues the L2c numbering
    device : "cuda" on a GPU node, "cpu" for a smoke test
    dry_run : if True, only print the commands without executing them
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    training = cfg.get("training", {})

    if seeds is None:
        seeds = [3, 4, 5][:n_models]
    if len(seeds) != n_models:
        raise ValueError(f"n_models={n_models} but got {len(seeds)} seeds: {seeds}")

    # Zielverzeichnis: neue Member landen neben den vorhandenen unter models/.
    ens_dir = cfg.get("pretrained_ensembles", {}).get(
        architecture, f"models/ensemble_{architecture}"
    )
    model_dir = (PROJECT_ROOT / ens_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = (PROJECT_ROOT / "logs" / f"train_ensemble_{architecture}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(rel):
        return str((PROJECT_ROOT / rel).resolve()) if rel else None

    train_file = _resolve(training.get("data_path", "data/raw/water_train.xyz"))
    if train_file is None:  # data_path fehlt in der config -> harter Fehler statt None
        raise ValueError("training.data_path fehlt in der config")
    valid_file = _resolve(training.get("valid_path"))  # optional
    test_file = _resolve(training.get("test_path_small", "data/raw/water_test_small.xyz"))

    # Hyperparameter: config-model-Block ueberschreibt die L2c-Defaults.
    hparams = dict(L2C_DEFAULTS)
    for k, v in (cfg.get("model") or {}).items():
        if k in ("architecture", "l_max"):
            continue
        hparams[k] = v
    if (cfg.get("model") or {}).get("l_max") is not None:
        hparams["max_L"] = cfg["model"]["l_max"]

    output_paths: list[str] = []
    for seed in seeds:
        # Namensschema wie vorhandene Checkpoints: mace-L2-c-03, -04, -05
        name = f"mace-L2-c-{seed:02d}"
        cmd = _build_train_command(
            name=name,
            seed=seed,
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
            model_dir=model_dir,
            log_dir=log_dir,
            device=device,
            max_epochs=training.get("max_epochs", 200),
            batch_size=training.get("batch_size", 8),
            learning_rate=training.get("learning_rate", 0.001),
            hparams=hparams,
        )
        output_paths.append(str(model_dir / f"{name}.model"))

        print(f"\n=== Training member seed={seed} -> {name}.model ===")
        print(" ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)

    print(f"\nDone. {len(seeds)} member(s) written to {model_dir}")
    return output_paths
