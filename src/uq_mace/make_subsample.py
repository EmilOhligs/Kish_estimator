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

def _repo_wurzel() -> Path:
    """Verzeichnis mit cache/ suchen -- das Skript liegt in src/uq_mace/,
    nicht mehr im Wurzelverzeichnis."""
    p = Path(__file__).resolve()
    for kandidat in p.parents:
        if (kandidat / "cache").is_dir():
            return kandidat
    raise SystemExit("kein cache/-Verzeichnis oberhalb von " + str(p))


ROOT = _repo_wurzel()
CACHE = ROOT / "cache"

N_TOTAL = 14811                                   # Frames in results_all_corrected.xyz
L0_MODELS = ["mace-L0-01", "mace-L0-c-01"]        # schon voll gerechnet (…_testfull.npz)


def pruefe_konsistenz(n: int) -> bool:
    """Beschreiben alle vorhandenen Subsample-Caches dieselben Frames?

    Die Frame-Gleichheit ruht auf zwei Annahmen, die sonst nirgends geprueft
    werden: dass Zeile j des vollen Caches Frame j der Quelldatei ist (fuer den
    Slicing-Pfad hier), und dass compute_full_energies --indices dieselben
    Frames liest (fuer den Neurechnungs-Pfad von L2). Zwei Wege, die sich
    treffen muessen.

    Pruefbar ist das ueber e_dft: die DFT-Energie eines Frames haengt nicht vom
    MACE-Modell ab. Beschreiben zwei Dateien dieselben Frames, sind ihre
    e_dft-Arrays BITGLEICH. Bei einer Verschiebung um einen Frame stuende dort
    ein kleiner, aber von null verschiedener Wert.
    """
    dateien = sorted(CACHE.glob(f"*_testfull_n{n}.npz"))
    if len(dateien) < 2:
        print(f"\n[Pruefung] weniger als zwei Subsample-Caches -- uebersprungen")
        return True

    print(f"\n[Pruefung] beschreiben alle {len(dateien)} Caches dieselben Frames?")
    ref_datei, ref = None, None
    ref_idx = None
    ok = True
    for p in dateien:
        d = np.load(p)
        e_dft, idx = d["e_dft"], d.get("idx")
        if ref is None:
            ref_datei, ref, ref_idx = p.name, e_dft, idx
            print(f"  {p.name:<40} n={e_dft.size}   (Referenz)")
            continue
        gleich_e = e_dft.shape == ref.shape and np.array_equal(e_dft, ref)
        gleich_i = idx is not None and ref_idx is not None and np.array_equal(idx, ref_idx)
        marke = "OK" if gleich_e and gleich_i else "ABWEICHUNG"
        print(f"  {p.name:<40} n={e_dft.size}   e_dft {'==' if gleich_e else '!='} "
              f"Referenz, idx {'==' if gleich_i else '!='} Referenz   {marke}")
        if not (gleich_e and gleich_i):
            ok = False
            if e_dft.shape == ref.shape:
                print(f"      max |e_dft-Diff| = {np.max(np.abs(e_dft - ref)):.3e}")
    if ok:
        print(f"  -> alle Caches liegen auf denselben {ref.size} Strukturen "
              f"(Referenz: {ref_datei})")
    else:
        print("  -> ACHTUNG: die Dateien beschreiben NICHT dieselben Frames. "
              "Jeder Modellvergleich darauf waere hinfaellig.")
    return ok


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

    if not pruefe_konsistenz(n):
        raise SystemExit(1)

    print("\nNaechster Schritt — L2 auf denselben Indizes:")
    print(f"  python compute_full_energies.py cpu --indices {idx_file} mace-L2-c-01")
    print("  danach dieses Skript erneut aufrufen, dann prueft es auch L2 mit.")


if __name__ == "__main__":
    main()
