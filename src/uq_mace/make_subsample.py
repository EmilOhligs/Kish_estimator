#!/usr/bin/env python
"""
Feste Frame-Teilmenge fuer den modelluebergreifenden Vergleich erzeugen.
============================================================================

Waehlt EINMAL n gleichmaessig ueber den grossen Datensatz verteilte Frame-Indizes
und legt sie ab. Aus den bereits gerechneten VOLLEN L0-Caches werden per Slicing
die passenden Subsamples gebaut (kein Neurechnen). L2 wird spaeter mit

    python compute_full_energies.py cpu --indices cache/subsample_n5000_idx.npy mace-L2-c-01

auf GENAU denselben Indizes ausgewertet -> alle Modelle auf identischen Strukturen.

Auswahl: np.linspace + astype(int) + unique = gleichmaessig verteilt, deterministisch,
deckt die ganze Datei ab (Frames 0 .. N-1). Fuer iid-Daten repraesentativ.

    python make_subsample.py            # n = 5000 (Default)
    python make_subsample.py 3000       # andere Groesse
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"

N_TOTAL = 14811                                   # Frames in results_all_corrected.xyz
L0_MODELS = ["mace-L0-01", "mace-L0-c-01"]        # schon voll gerechnet (…_testfull.npz)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    # --- gleichmaessig verteilte Indizes ueber die GANZE Datei ---
    idx = np.unique(np.linspace(0, N_TOTAL - 1, n).astype(int))
    idx_file = CACHE / f"subsample_n{n}_idx.npy"
    np.save(idx_file, idx)
    print(f"{len(idx)} Indizes (Bereich {idx.min()}..{idx.max()}) -> {idx_file.name}")

    # --- L0-Subsamples durch Slicing der vollen Caches (Zeile j = Frame j) ---
    for name in L0_MODELS:
        full = CACHE / f"single_{name}_testfull.npz"
        if not full.exists():
            print(f"[FEHLT] {full.name} — L0 zuerst voll rechnen"); continue
        d = np.load(full)
        out = CACHE / f"single_{name}_testfull_n{n}.npz"
        np.savez(out, idx=idx, e_dft=d["e_dft"][idx], e_mace=d["e_mace"][idx])
        print(f"  {out.name}: {len(idx)} Frames")

    print("\nNaechster Schritt — L2 auf denselben Indizes:")
    print(f"  python compute_full_energies.py cpu --indices {idx_file} mace-L2-c-01")


if __name__ == "__main__":
    main()
