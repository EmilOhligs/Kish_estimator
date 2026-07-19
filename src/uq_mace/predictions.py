"""Gemeinsame MACE-Ensemble-Vorhersagen mit Platten-Cache.

Laesst jedes Ensemble-Member EINMAL ueber einen Testsatz laufen und cacht die
per-Frame-Energien und per-Atom-Kraefte nach

    results/predictions_<ensemble>_test<testset>.npz

So greifen alle nachgelagerten Analysen (Energy/Force-RMSE, sigma-Fehler-
Korrelation, Reweighting-Gewichte w_i) auf IDENTISCHE Zahlen zu, und die teure
MACE-Inferenz laeuft nur ein einziges Mal pro (Ensemble, Testsatz).

Verwendung:
    from uq_mace.predictions import get_predictions
    pred = get_predictions("ensemble_L2c", "big")
    e_dft   = pred["e_dft"]        # (F,)      DFT-Gesamtenergie pro Frame
    energies= pred["energies"]     # (M, F)    Energie je Member
    e_mace  = energies.mean(0)     # (F,)      Ensemble-Mittel
    forces  = pred["forces"]       # Liste F x (M, n_i, 3)
    f_ref   = pred["f_ref"]        # Liste F x (n_i, 3)
    n_atoms = pred["n_atoms"]      # (F,)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import TEST_SET_BIG, TEST_SET_SMALL, load_trajectory
from .ensemble import MODELS_DIR
from .evaluation import evaluate_ensemble_members, reference_energies_forces

TEST_SETS = {"big": TEST_SET_BIG, "small": TEST_SET_SMALL}
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def cache_path(ensemble: str, testset: str) -> Path:
    return RESULTS_DIR / f"predictions_{ensemble}_test{testset}.npz"


def _to_object_array(seq) -> np.ndarray:
    """1D-Object-Array aus einer Liste (ggf. gleichgeformter) Arrays.

    np.array(list_of_arrays, dtype=object) wuerde bei einheitlicher Atomzahl ein
    3D-Array bauen statt eines 1D-Arrays von Arrays -> hier explizit fuellen.
    """
    arr = np.empty(len(seq), dtype=object)
    for i, x in enumerate(seq):
        arr[i] = np.asarray(x)
    return arr


K_B = 8.617333262e-5  # eV/K


def load_energies(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(e_dft, e_mace) aus einem Energie-Cache lesen - ohne MACE-Neulauf.

    Akzeptiert sowohl den aktuellen predictions_*.npz (Keys 'e_dft','energies')
    als auch aeltere mace_energies_*.npz (zusaetzlich 'e_mace'). e_mace ist das
    Ensemble-Mittel.
    """
    p = Path(path)
    d = np.load(p, allow_pickle=True)
    if "e_dft" not in d.files:
        raise KeyError(f"{p.name}: kein 'e_dft' enthalten (Keys: {d.files})")
    e_dft = np.asarray(d["e_dft"], dtype=float)
    if "e_mace" in d.files:
        e_mace = np.asarray(d["e_mace"], dtype=float)
    elif "energies" in d.files:
        e_mace = np.asarray(d["energies"], dtype=float).mean(axis=0)
    else:
        raise KeyError(f"{p.name}: 'e_dft' ohne 'e_mace'/'energies'")
    return e_dft, e_mace


def load_weights(path: str | Path, temperature: float = 300.0) -> np.ndarray:
    """Gewichte w_i aus einer vorhandenen Datei beschaffen - ohne MACE-Neulauf.

    Erkennt automatisch:
      * fertige Gewichte : .npz-Key 'w'/'weights', oder .npy/.txt/.csv-Array
      * Energie-Cache    : .npz mit 'e_dft' und ('e_mace' oder 'energies')
                           -> w = exp(-beta*(e_dft - e_mace)) wird hier berechnet
                              (beta = 1/(k_B T)). Deckt sowohl den aktuellen
                              predictions_*.npz als auch aeltere mace_energies_*.npz ab.
    """
    from .reweighting import reweighting_weights

    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        if any(k in d.files for k in ("w", "weights")):
            key = "w" if "w" in d.files else "weights"
            w = d[key]
            print(f"[load ] {np.asarray(w).size} fertige Gewichte aus {p.name}")
        elif "e_dft" in d.files:
            e_dft, e_mace = load_energies(p)
            beta = 1.0 / (K_B * temperature)
            w = reweighting_weights(e_dft, e_mace, beta)
            print(f"[calc ] w_i aus e_dft/e_mace in {p.name} berechnet "
                  f"(T={temperature:.0f} K, beta={beta:.2f} eV^-1), n={w.size}")
        else:
            w = d[d.files[0]]
            print(f"[load ] {np.asarray(w).size} Werte aus {p.name} (Key '{d.files[0]}')")
    elif p.suffix == ".npy":
        w = np.load(p)
        print(f"[load ] {np.asarray(w).size} Gewichte aus {p.name}")
    else:  # .txt / .csv / .dat
        w = np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None)
        print(f"[load ] {np.asarray(w).size} Gewichte aus {p.name}")
    return np.asarray(w, dtype=float).ravel()


def get_predictions(
    ensemble: str = "ensemble_L2c",
    testset: str = "big",
    *,
    force: bool = False,
    device: str = "cpu",
) -> dict:
    """MACE-Vorhersagen fuer (ensemble, testset), mit Cache.

    Gibt ein dict zurueck mit:
        energies : (M, F)                 Gesamtenergie je Member und Frame [eV]
        forces   : Liste F x (M, n_i, 3)  Kraefte je Member [eV/A]
        e_dft    : (F,)                   DFT-Referenzenergie je Frame [eV]
        f_ref    : Liste F x (n_i, 3)     DFT-Referenzkraefte [eV/A]
        n_atoms  : (F,)                   Atomzahl je Frame

    force=True erzwingt Neuberechnung (Cache wird ueberschrieben).
    """
    cache = cache_path(ensemble, testset)
    if cache.exists() and not force:
        print(f"[cache] lade {cache.name}")
        d = np.load(cache, allow_pickle=True)
        return dict(
            energies=d["energies"],
            forces=list(d["forces"]),
            e_dft=d["e_dft"],
            f_ref=list(d["f_ref"]),
            n_atoms=d["n_atoms"],
        )

    frames = load_trajectory(TEST_SETS[testset])
    print(f"[eval ] {len(frames)} Frames aus {TEST_SETS[testset].name}")

    # WICHTIG: DFT-Referenz VOR dem Anhaengen der Calculator auslesen.
    e_dft, f_ref, n_atoms = reference_energies_forces(frames)

    model_dir = MODELS_DIR / ensemble
    n_member = len(list(model_dir.glob("*.model")))
    print(f"[eval ] Ensemble {ensemble} ({n_member} Member) auf {device} ...")
    result = evaluate_ensemble_members(model_dir, frames, device=device)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        energies=result["energies"],
        forces=_to_object_array(result["forces"]),
        e_dft=e_dft,
        f_ref=_to_object_array(f_ref),
        n_atoms=n_atoms,
    )
    print(f"[cache] gespeichert -> {cache.name}")
    return dict(
        energies=result["energies"],
        forces=result["forces"],
        e_dft=e_dft,
        f_ref=f_ref,
        n_atoms=n_atoms,
    )
