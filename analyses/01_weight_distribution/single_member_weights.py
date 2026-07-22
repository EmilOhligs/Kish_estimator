"""
ΔE- und w_i-Verteilung für je EIN Ensemble-Member (L0 und L0c)
============================================================================

Bewusst einfach: nimmt je einen einzelnen MACE-Checkpoint (nicht das
Ensemble-Mittel), laesst ihn ueber water_testset_big.xyz laufen und zeigt

    dE_i = E_DFT(R_i) - E_MACE(R_i)          und      w_i = exp(-beta*dE_i)

Ein einzelnes Member entspricht dem realen Reweighting-Fall naeher als der
Ensemble-Mittelwert - im Paper wird mit EINEM Potential reweightet.

Kosten: 1 Modell x 400 Frames je Ensemble (statt 3 beim Ensemble-Mittel).

Ausfuehren:
    python analyses/01_weight_distribution/single_member_weights.py
    python analyses/01_weight_distribution/single_member_weights.py --device mps
    python analyses/01_weight_distribution/single_member_weights.py \
        --models models/ensemble_L0/mace-L0-01.model models/ensemble_L2c/mace-L2-c-01.model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

from uq_mace.data import TEST_SET_BIG, TEST_SET_SMALL, load_trajectory
from uq_mace.reweighting import (
    effective_sample_size,
    psis_khat,
    reweighting_weights,
)

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
K_B = 8.617333262e-5  # eV/K
TEST_SETS = {"big": TEST_SET_BIG, "small": TEST_SET_SMALL}


def energies_for_model(model_path: Path, testset: str, device: str, force: bool):
    """(e_dft, e_mace) fuer EIN Modell; Ergebnis wird gecacht."""
    cache = CACHE / f"single_{model_path.stem}_test{testset}.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        print(f"[cache] {cache.name}")
        return d["e_dft"], d["e_mace"]

    from mace.calculators import MACECalculator

    frames = load_trajectory(TEST_SETS[testset])
    e_dft = np.array([f.get_potential_energy() for f in frames])   # vor dem Calculator!

    print(f"[eval ] {model_path.name} ueber {len(frames)} Frames auf {device} ...")
    calc = MACECalculator(model_paths=[str(model_path)], device=device)
    e_mace = np.empty(len(frames))
    for i, atoms in enumerate(frames):
        atoms.calc = calc
        e_mace[i] = atoms.get_potential_energy()

    CACHE.mkdir(exist_ok=True)
    np.savez(cache, e_dft=e_dft, e_mace=e_mace)
    print(f"[cache] gespeichert -> {cache.name}")
    return e_dft, e_mace


def stats(e_dft, e_mace, beta):
    dE = e_dft - e_mace
    w = reweighting_weights(e_dft, e_mace, beta)
    n = dE.size
    neff = effective_sample_size(w)
    return dict(dE=dE, w=w, n=n,
                std_meV=dE.std(ddof=1) * 1000,
                c=beta * dE.std(ddof=1),
                gamma1=float(skew(dE)), gamma2=float(kurtosis(dE)),
                cv=float(w.std() / w.mean()),
                neff=neff, ratio=neff / n, khat=psis_khat(w))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Pfade zu .model-Dateien (Default: je erstes Member von L0 und L0c)")
    ap.add_argument("--testset", default="big", choices=["big", "small"])
    ap.add_argument("--temperature", type=float, default=292.0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--force", action="store_true", help="Cache ignorieren")
    args = ap.parse_args()

    if args.models:
        models = [Path(p) for p in args.models]
    else:
        models = [sorted((ROOT / "models" / e).glob("*.model"))[0]
                  for e in ("ensemble_L0", "ensemble_L0c")]

    beta = 1.0 / (K_B * args.temperature)
    results = []
    for mp in models:
        if not mp.exists():
            print(f"  {mp}: nicht gefunden, uebersprungen")
            continue
        e_dft, e_mace = energies_for_model(mp, args.testset, args.device, args.force)
        r = stats(e_dft, e_mace, beta)
        r["name"] = mp.stem
        results.append(r)

    if not results:
        raise SystemExit("Kein Modell auswertbar.")

    # ---- Tabelle ----
    print("\n" + "=" * 92)
    print(f"{'Modell':<18}{'std(dE)':>10}{'c':>8}{'gamma1':>9}{'gamma2':>9}"
          f"{'CV(w)':>8}{'N_eff/n':>9}{'e^-c^2':>9}{'khat':>8}")
    print(f"{'':18}{'[meV]':>10}{'':>8}{'':>9}{'':>9}{'':>8}{'':>9}{'Gauss':>9}{'':>8}")
    print("-" * 92)
    for r in results:
        print(f"{r['name']:<18}{r['std_meV']:>10.2f}{r['c']:>8.3f}{r['gamma1']:>+9.3f}"
              f"{r['gamma2']:>+9.3f}{r['cv']:>8.3f}{r['ratio']:>9.3f}"
              f"{np.exp(-r['c']**2):>9.3f}{r['khat']:>8.3f}")
    print("=" * 92)

    # ---- Plot: je Modell eine Zeile, links dE, rechts w ----
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 4.2 * len(results)),
                             squeeze=False)
    for row, r in enumerate(results):
        a, b = axes[row]
        dEc = (r["dE"] - r["dE"].mean()) * 1000
        a.hist(dEc, bins=40, density=True, alpha=0.65, color="steelblue")
        xs = np.linspace(dEc.min(), dEc.max(), 200)
        a.plot(xs, np.exp(-0.5 * (xs / r["std_meV"]) ** 2)
               / (r["std_meV"] * np.sqrt(2 * np.pi)), "r--", lw=1.4, label="Gauß-Fit")
        a.set_xlabel("$\\Delta E$ (zentriert) [meV/Frame]")
        a.set_ylabel("Dichte")
        a.set_title(f"{r['name']} — $\\Delta E$   "
                    f"(std {r['std_meV']:.1f} meV, $\\gamma_1$={r['gamma1']:+.2f})")
        a.legend(fontsize=8.5); a.grid(alpha=0.3)

        wn = r["w"] / r["w"].mean()
        b.hist(wn, bins=45, density=True, alpha=0.65, color="darkorange")
        b.axvline(1.0, color="k", ls=":", lw=1)
        b.set_yscale("log")
        b.set_xlabel("$w_i / \\langle w\\rangle$")
        b.set_ylabel("Dichte (log)")
        b.set_title(f"{r['name']} — Gewichte   "
                    f"(c={r['c']:.3f}, $N_{{eff}}/n$={r['ratio']:.3f}, "
                    f"$\\hat k$={r['khat']:.2f})")
        b.grid(alpha=0.3, which="both")

    fig.suptitle(f"Einzel-Member: $\\Delta E$ und Gewichte   |   "
                 f"test{args.testset}, T = {args.temperature:.0f} K", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / f"single_member_weights_test{args.testset}.png"
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] {out}")


if __name__ == "__main__":
    main()
