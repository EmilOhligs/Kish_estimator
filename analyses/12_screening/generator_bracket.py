"""
Screening ohne Kenntnis des Generators — Klammer statt Punktwert
============================================================================

DAS PROBLEM. Die Ensemble-Korrektur braucht p_test, also das Potential, mit dem
die Testkonfigurationen erzeugt wurden. Steht das nicht fest, ist die Korrektur
nicht bestimmt:

    p_MACE / p_test  ∝  exp(-beta (E_MACE - E_test))     <- E_test unbekannt

Vollstaendig generatoragnostisch geht daher NICHT. Kein Schaetzer kann eine
Verteilung korrigieren, die er nicht kennt.

WAS STATTDESSEN GEHT. Ueber eine FAMILIE plausibler Generatoren rechnen und ein
INTERVALL ausgeben. Die Familie hier:

    * AIMD-Grenzfall        p_test = p_DFT
      -> Transfer ueber dE_MACE selbst, exakt per Rueckwaerts-Umgewichtung
    * jedes verfuegbare MLIP im Cache
      -> Transfer ueber E_MACE - E_X, exakt, OHNE DFT
      (Die Frames stammen real nicht aus p_X; die Rechnung beantwortet
       "was WAERE, wenn X der Generator gewesen waere". Genau das ist der
       Zweck einer Sensitivitaetsanalyse.)

Ausgegeben wird [min, max] ueber die Familie, plus die Breite als
Robustheitsmass. Ist die Klammer schmal, ist die Unkenntnis des Generators
folgenlos — und DAS ist auf den vorliegenden Daten pruefbar, ohne neue Messung.

WAS DAMIT NICHT GEPRUEFT IST.
  * Ob der wahre Generator in der Familie liegt. Ein Modell ausserhalb (etwa
    ein klassisches Kraftfeld, weit von DFT) faellt aus der Klammer.
  * Abdeckung (A3). Die Umgewichtung gewichtet um, was da ist; sie kann keine
    Konfiguration erzeugen, die im Testsatz fehlt. Siehe
    validate_ensemble_shift.py, Teil 3.
  * Die absolute Korrektheit. Dafuer fehlt auf realen Daten die Wahrheit —
    deshalb laeuft die Korrektheitspruefung am Gamma-Modell.

WICHTIGER SONDERFALL. Ist der Generator dasselbe (oder ein sehr aehnliches)
Modell wie das bewertete, verschwindet die Korrektur — man gewichtet dann von
p_MACE nach p_MACE. Das Skript weist diesen Fall ueber die Fehlerkorrelation rho
aus: rho nahe 1 heisst "gleiche Architektur, Korrektur ~ 0".

Ausfuehren:
    python analyses/12_screening/generator_bracket.py
    python analyses/12_screening/generator_bracket.py --model mace-L0-01
    python analyses/12_screening/generator_bracket.py --temperature 298
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.predictions import load_energies

HERE = Path(__file__).resolve().parent
CACHE = Path(__file__).resolve().parents[2] / "cache"
K_B = 8.617333262e-5  # eV/K


def tidy(stem: str) -> str:
    for a, b in (("single_", ""), ("mace_energies_", ""), ("predictions_", ""),
                 ("_testbig", ""), ("_testsmall", "")):
        stem = stem.replace(a, b)
    return stem


def _lse(v: np.ndarray) -> float:
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()))


def weighted_stats(dE: np.ndarray, log_v: np.ndarray, beta: float) -> dict:
    """Statistik von dE unter der durch log_v gegebenen Umgewichtung."""
    v = np.exp(log_v - log_v.max())
    v = v / v.sum()
    mean = float((v * dE).sum())
    var = float((v * (dE - mean) ** 2).sum())
    sd = np.sqrt(max(var, 0.0))
    g1 = float((v * (dE - mean) ** 3).sum() / sd ** 3) if sd > 0 else 0.0
    g2 = float((v * (dE - mean) ** 4).sum() / sd ** 4 - 3.0) if sd > 0 else 0.0

    # N_eff/n unter der Zielverteilung: E[w]^2 / E[w^2], beide unter v gewichtet
    lw = -beta * dE
    lv = np.log(v)
    e_w = _lse(lv + lw)
    e_w2 = _lse(lv + 2 * lw)
    ratio = float(np.exp(2 * e_w - e_w2))

    return dict(c=float(beta * sd), gamma1=g1, gamma2=g2, ratio=ratio,
                neff_transfer=float(1.0 / (v ** 2).sum()))


def bootstrap_ratio(dE, log_v, beta, reps=400, rng=None) -> float:
    """Standardfehler von N_eff/n unter der Umgewichtung, per Bootstrap.

    Noetig, weil die Transfergewichte selbst degeneriert sein koennen: bei
    N_eff(Transfer) ~ 150/400 traegt nur ein Bruchteil der Frames, und die
    Streuung der Schaetzung ist dann NICHT vernachlaessigbar gegen die
    Unterschiede zwischen Generatoren. Ohne diese Zahl verwechselt man
    Schaetzrauschen mit echter Generator-Abhaengigkeit.
    """
    rng = rng or np.random.default_rng(0)
    n = dE.size
    out = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        out[i] = weighted_stats(dE[idx], log_v[idx], beta)["ratio"]
    return float(out.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ensemble_L2c",
                    help="das zu screenende Modell (Cache-Name)")
    ap.add_argument("--temperature", type=float, default=292.0)
    ap.add_argument("--exclude-rho", type=float, default=0.8,
                    help="Generatoren mit Fehlerkorrelation ueber diesem Wert aus der "
                         "Klammer nehmen — sie sind derselbe Modelltyp. Nur setzen, "
                         "wenn BEKANNT ist, dass der Testsatz nicht damit erzeugt wurde.")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    beta = 1.0 / (K_B * args.temperature)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- alle Caches einlesen ---
    pool, seen = {}, set()
    for p in sorted(CACHE.glob("*.npz")):
        try:
            if "e_dft" not in np.load(p, allow_pickle=True).files:
                continue
        except Exception:
            continue
        name = tidy(p.stem)
        if name in seen:
            continue
        seen.add(name)
        pool[name] = load_energies(p)

    if args.model not in pool:
        raise SystemExit(f"{args.model} nicht im Cache. Vorhanden: {sorted(pool)}")

    e_dft, e_M = pool[args.model]
    dE = e_dft - e_M
    n = dE.size

    rows = []

    def add(gen, rho, log_v, kind):
        r = weighted_stats(dE, log_v, beta)
        r.update(gen=gen, rho=rho, kind=kind,
                 se=bootstrap_ratio(dE, log_v, beta, args.bootstrap, rng))
        rows.append(r)

    # (1) unkorrigiert: Testsatz-Wert, gilt nur falls p_test = p_MACE
    add("— unkorrigiert (p_test = p_MACE) —", np.nan, np.zeros(n), "ref")

    # (2) AIMD-Grenzfall: p_test = p_DFT  ->  Transfer mit 1/w = exp(+beta dE)
    add("AIMD  (p_test = p_DFT)", 0.0, beta * (dE - dE.mean()), "family")

    # (3) jedes andere MLIP als hypothetischer Generator
    for name, (_, e_X) in pool.items():
        if name == args.model:
            continue
        diff = e_M - e_X                       # Modell gegen Modell, KEINE DFT
        rho = float(np.corrcoef(dE, e_dft - e_X)[0, 1])
        kind = "excluded" if rho > args.exclude_rho else "family"
        add(name, rho, -beta * (diff - diff.mean()), kind)

    # --- Tabelle ---
    print(f"\nSCREENING MIT UNBEKANNTEM GENERATOR — {args.model}, T = {args.temperature:.0f} K")
    print("=" * 100)
    print(f"{'angenommener Generator':<30}{'rho':>8}{'c_prod':>9}"
          f"{'N_eff/n':>20}{'N_eff(Transfer)':>16}{'':>6}")
    print("-" * 100)
    for r in rows:
        rho = "  —  " if np.isnan(r["rho"]) else f"{r['rho']:+.3f}"
        tag = {"ref": "", "family": "", "excluded": "  (ausgeschl.)"}[r["kind"]]
        print(f"{r['gen']:<30}{rho:>8}{r['c']:>9.4f}"
              f"{r['ratio']:>13.4f} ± {r['se']:.4f}"
              f"{r['neff_transfer']:>12.0f}/{n:<4d}{tag}")
    print("=" * 100)
    print(f"  ausgeschlossen: rho > {args.exclude_rho:.2f} — derselbe Modelltyp wie das")
    print(f"  bewertete Modell. Gilt nur, weil BEKANNT ist, dass der Testsatz nicht")
    print(f"  mit einem weiteren MACE-Modell erzeugt wurde.")

    # --- Klammer ---
    fam = [r for r in rows if r["kind"] == "family"]
    lo = min(r["ratio"] for r in fam)
    hi = max(r["ratio"] for r in fam)
    uncorr = rows[0]["ratio"]
    mid = 0.5 * (lo + hi)
    corr = mid - uncorr
    se_typ = float(np.mean([r["se"] for r in fam]))

    print(f"\nKLAMMER ueber die Generator-Familie:  N_eff/n = {lo:.4f} ... {hi:.4f}")
    print(f"  Breite der Klammer               {100*(hi-lo):.2f} Prozentpunkte")
    print(f"  typischer Bootstrap-SE je Punkt  {100*se_typ:.2f} Prozentpunkte")
    print(f"  unkorrigierter Testsatz-Wert     {uncorr:.4f}")
    print(f"  Korrektur (Klammermitte)         {100*corr:+.2f} Prozentpunkte")

    print(f"\n  Ist die Klammerbreite ECHTE Generator-Abhaengigkeit oder Schaetzrauschen?")
    print(f"    Breite / (2*SE) = {(hi-lo)/(2*se_typ):.2f}   "
          f"{'-> ueberwiegend RAUSCHEN' if (hi-lo) < 2*se_typ else '-> echte Abhaengigkeit'}")
    print(f"\n  Ist die Unkenntnis des Generators folgenlos?")
    print(f"    Breite / |Korrektur| = {(hi-lo)/abs(corr):.2f}   "
          f"{'-> ja' if (hi-lo) < 0.5*abs(corr) else '-> nein, Generator muss bekannt sein'}")

    print("\n  Vorbehalte: (a) der wahre Generator muss in der Familie liegen;")
    print("              (b) Abdeckung (A3) ist hiermit NICHT geprueft;")
    print("              (c) Generatoren mit rho nahe 1 sind derselbe Modelltyp —")
    print("                  ausschliessen, sobald bekannt ist, dass der Testsatz")
    print("                  nicht mit diesem Modell erzeugt wurde.")

    # --- CSV ---
    f1 = args.outdir / f"generator_bracket_{args.model}.csv"
    with open(f1, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["assumed_generator", "rho", "c_prod", "gamma1", "gamma2",
                     "neff_ratio", "neff_transfer", "n"])
        for r in rows:
            wr.writerow([r["gen"], "" if np.isnan(r["rho"]) else f"{r['rho']:.4f}",
                         f"{r['c']:.5f}", f"{r['gamma1']:.4f}", f"{r['gamma2']:.4f}",
                         f"{r['ratio']:.5f}", f"{r['neff_transfer']:.1f}", n])

    # --- Plot ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))

    lbl = [r["gen"] for r in rows]
    val = [r["ratio"] for r in rows]
    y = np.arange(len(rows))
    cols = ["gray" if r["kind"]=="ref" else ("lightgray" if r["kind"]=="excluded" else "steelblue") for r in rows]
    a1.barh(y, val, color=cols, height=0.6)
    a1.axvspan(lo, hi, color="orange", alpha=0.20, zorder=0,
               label=f"Klammer {lo:.3f}–{hi:.3f}")
    a1.axvline(uncorr, color="crimson", ls="--", lw=1.6,
               label=f"unkorrigiert = {uncorr:.4f}")
    a1.set_yticks(y)
    a1.set_yticklabels(lbl, fontsize=8.5)
    a1.set_xlim(min(val) - 0.02, max(val) + 0.01)
    a1.set_xlabel("$N_\\mathrm{eff}/n$ im Produktionslauf")
    a1.set_title("(a) Prognose je angenommenem Generator")
    a1.legend(fontsize=8.5, loc="lower right")
    a1.grid(alpha=0.3, axis="x")

    rr = [r["rho"] for r in rows if not np.isnan(r["rho"])]
    vv = [r["ratio"] for r in rows if not np.isnan(r["rho"])]
    nn = [r["gen"] for r in rows if not np.isnan(r["rho"])]
    a2.scatter(rr, vv, s=90, color="steelblue", zorder=3)
    for x, yv, t in zip(rr, vv, nn):
        a2.annotate(t, (x, yv), textcoords="offset points", xytext=(6, 6), fontsize=8)
    a2.axhline(uncorr, color="crimson", ls="--", lw=1.5, label="unkorrigiert")
    a2.set_xlabel("$\\rho$ — Fehlerkorrelation Generator vs. bewertetes Modell")
    a2.set_ylabel("$N_\\mathrm{eff}/n$")
    a2.set_title("(b) je ähnlicher der Generator, desto kleiner die Korrektur")
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3)

    fig.suptitle(
        f"Screening bei unbekanntem Generator — {args.model}, T = {args.temperature:.0f} K\n"
        f"Klammer statt Punktwert; Transfer über $E_\\mathrm{{MACE}}-E_X$ exakt und ohne DFT",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = args.outdir / f"generator_bracket_{args.model}.png"
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] {out}")
    print(f"[csv  ] {f1}")


if __name__ == "__main__":
    main()
