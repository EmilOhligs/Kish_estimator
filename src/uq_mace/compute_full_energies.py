#!/usr/bin/env python
"""
MACE-Energien fuer den grossen Datensatz rechnen und cachen (Overnight-Lauf).
============================================================================

Rechnet fuer jedes Modell die MACE-Gesamtenergie ueber alle Frames von
data/raw/results_all_corrected.xyz (14811 Strukturen, 63 H2O) und legt sie neben
der DFT-Referenz als Cache ab — EINE eigene Datei pro Modell:

    cache/single_<modell>_testfull.npz        (Keys: e_dft, e_mace)

Nur ENERGIEN (keine Kraefte) -> schnell und klein. Fuer den ΔE/N_eff-Workflow
ist mehr nicht noetig. Ein Ensemble-Mittel (z.B. der L2-c-Member) wird NICHT hier
gebildet, sondern bei Bedarf im Notebook aus den einzelnen Modell-Dateien.

ROBUST fuer einen Lauf ueber Nacht:
  * STREAMING (ase.io.iread) -> haelt nie die ganze 287-MB-Datei im Speicher.
  * CHECKPOINT alle 2000 Frames in eine .partial.npz -> ein Absturz verliert
    hoechstens die letzten Frames; ein Neustart setzt automatisch fort.
  * SKIP: ist der finale Cache schon da, wird das Modell uebersprungen.
  * FORTSCHRITT mit Rate und ETA.

Aufruf (im aktivierten venv, vom Projekt-Root):
    python compute_full_energies.py                 # cpu, alle 4 Modelle
    python compute_full_energies.py mps             # Apple-GPU (falls MACE es unterstuetzt)
    python compute_full_energies.py cpu mace-L0-01 mace-L0-c-01   # nur diese Modelle
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "results_all_corrected.xyz"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

N_TOTAL = 14811          # bekannte Frameanzahl (nur fuer die ETA-Anzeige)
CKPT_EVERY = 2000
LOG_EVERY = 200

# name -> .model-Pfad
ALL_MODELS = {
    "mace-L0-01":   ROOT / "models/ensemble_L0/mace-L0-01.model",
    "mace-L0-c-01": ROOT / "models/ensemble_L0c/mace-L0-c-01.model",
    "mace-L2-c-01": ROOT / "models/ensemble_L2c/mace-L2-c-01.model",
    "mace-L2-c-02": ROOT / "models/ensemble_L2c/mace-L2-c-02.model",
}


def run_model(name: str, path: Path, device: str, sel: np.ndarray, suffix: str) -> None:
    # sel = aufsteigende Quell-Indizes, die gerechnet werden sollen. Der Frame-INDEX
    # 'idx' wird mitgespeichert -> spaeter modelluebergreifend matchbar.
    sel_set = set(int(x) for x in sel)
    n_target = len(sel)

    final = CACHE / f"single_{name}_{suffix}.npz"
    if final.exists():
        print(f"[skip ] {name}: {final.name} existiert bereits", flush=True)
        return
    if not path.exists():
        print(f"[FEHLT] {name}: {path} nicht gefunden — uebersprungen", flush=True)
        return

    from ase.io import iread
    from mace.calculators import MACECalculator

    part = CACHE / f"single_{name}_{suffix}.partial.npz"
    idx: list[int] = []
    e_dft: list[float] = []
    e_mace: list[float] = []
    done = 0
    if part.exists():
        d = np.load(part)
        idx, e_dft, e_mace = list(d["idx"]), list(d["e_dft"]), list(d["e_mace"])
        done = len(e_mace)
        print(f"[resume] {name}: {done} Frames schon gerechnet, setze fort", flush=True)

    print(f"[load ] {name} auf '{device}', {n_target} von {N_TOTAL} Frames ...", flush=True)
    calc = MACECalculator(model_paths=[str(path)], device=device)

    t0 = time.time()
    seen = 0                                     # Zahl der bislang AUSGEWAEHLTEN Frames
    for i, atoms in enumerate(iread(str(DATA))):
        if i not in sel_set:                     # nicht ausgewaehlt -> ueberspringen
            continue
        if seen < done:                          # schon gerechnet (Resume) -> ueberspringen
            seen += 1
            continue
        ed = atoms.get_potential_energy()        # DFT (aus der Datei, VOR dem Calculator)
        atoms.calc = calc
        em = atoms.get_potential_energy()        # MACE
        idx.append(int(i))
        e_dft.append(float(ed))
        e_mace.append(float(em))
        seen += 1

        m = len(e_mace)
        if m % LOG_EVERY == 0:
            el = time.time() - t0
            rate = (m - done) / max(el, 1e-9)
            eta = (n_target - m) / max(rate, 1e-9) / 3600.0
            print(f"  {name}: {m}/{n_target}  ({rate:.1f} Frames/s, ETA {eta:.2f} h)",
                  flush=True)
        if m % CKPT_EVERY == 0:
            np.savez(part, idx=np.array(idx), e_dft=np.array(e_dft), e_mace=np.array(e_mace))
        if m >= n_target:
            break

    np.savez(final, idx=np.array(idx), e_dft=np.array(e_dft), e_mace=np.array(e_mace))
    try:
        part.unlink()
    except FileNotFoundError:
        pass
    print(f"[done ] {name}: {len(e_mace)} Frames -> {final.name}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


def main() -> None:
    args = sys.argv[1:]
    device = "cpu"
    limit: int | None = None
    idx_file: str | None = None
    if "--limit" in args:
        j = args.index("--limit"); limit = int(args[j + 1]); del args[j:j + 2]
    if "--indices" in args:
        j = args.index("--indices"); idx_file = args[j + 1]; del args[j:j + 2]
    if args and args[0] in ("cpu", "mps", "cuda"):
        device = args.pop(0)
    names = args if args else list(ALL_MODELS)

    if not DATA.exists():
        raise SystemExit(f"Datensatz nicht gefunden: {DATA}")

    # Auswahl der Frames: --indices (feste Menge aus Datei) > --limit (gleichverteilt) > alle
    if idx_file is not None:
        sel = np.load(idx_file).astype(int)
        suffix = f"testfull_n{len(sel)}"
        sel_info = f"aus {idx_file}"
    elif limit is not None:
        sel = np.unique(np.linspace(0, N_TOTAL - 1, min(limit, N_TOTAL)).astype(int))
        suffix = f"testfull_n{len(sel)}"
        sel_info = "gleichverteilt"
    else:
        sel = np.arange(N_TOTAL)
        suffix = "testfull"
        sel_info = "alle"

    print(f"Datensatz : {DATA.name}  ({N_TOTAL} Frames)")
    print(f"Device    : {device}")
    print(f"Frames    : {len(sel)}  ({sel_info})  -> Cache-Suffix '{suffix}'")
    print(f"Modelle   : {', '.join(names)}\n")

    t0 = time.time()
    for name in names:
        if name not in ALL_MODELS:
            print(f"[?] unbekanntes Modell '{name}' — bekannt: {list(ALL_MODELS)}")
            continue
        run_model(name, ALL_MODELS[name], device, sel, suffix)
    print(f"\nFertig. Gesamtzeit {(time.time()-t0)/3600:.2f} h.")


if __name__ == "__main__":
    main()
