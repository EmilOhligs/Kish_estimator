"""
Räumliche Korrelation des Kraftfehlers — prüft die √N-Annahme
============================================================================

DIE FRAGE: Die Extrapolation c ∝ sqrt(N) auf groessere Systeme steht auf einem
ungeprueften Glied. Die Kette lautet:

  1. dE ist extensiv (Summe lokaler Beitraege)              -> solide
  2. endliche Korrelationslaenge der Fehlerbeitraege        -> ??? HIER
  3. Summe ueber N/n_xi unabhaengige Bloecke -> Var ∝ N     -> folgt aus 2
  4. sigma ∝ sqrt(N), also c ∝ sqrt(N)                      -> folgt aus 3

Schritt 2 ist die Luecke. MACEs Rezeptivfeld (r_max=6 A x 2 Schichten ~ 12 A) macht
die Beitraege FUNKTIONAL lokal - aber daraus folgt NICHT, dass ihre Fehler
STATISTISCH unabhaengig sind. Ein systematischer Fehler in der Beschreibung eines
lokalen Motivs (z.B. Wasserstoffbruecken) erzeugt gleichgerichtete Fehler ueberall
im System; die addieren sich dann kohaerent und sigma ∝ N statt sqrt(N).

DER TEST: Anders als die Energie sind KRAEFTE pro Atom verfuegbar - aus DFT (in den
xyz-Dateien) und aus MACE. Damit laesst sich der Fehler pro Atom bilden

    dF_i = F_i(DFT) - F_i(MACE)

und seine raeumliche Korrelationsfunktion messen:

    C(r) = <dF_i . dF_j> / <|dF|^2>      fuer |r_i - r_j| = r

Faellt C(r) innerhalb weniger Angstroem auf die Grundlinie, sind die Fehler lokal
und die sqrt(N)-Annahme gestuetzt. Bleibt sie ueber die Zelle hinweg endlich, ist
sie widerlegt - mit vorhandenen Daten, ohne neue DFT-Rechnung.

ZWEI DINGE, DIE MAN WISSEN MUSS:

  * SUMMENREGEL. Die Gesamtkraft ist in beiden Rechnungen null, also sum_i dF_i = 0.
    Daraus folgt zwingend  <dF_i.dF_j>_{i!=j} = -<|dF|^2>/(N-1),  also eine
    eingebaute NEGATIVE Grundlinie bei -1/(N-1). C(r) laeuft also nicht gegen 0,
    sondern gegen diesen Wert. Er ist im Plot eingezeichnet.

  * REICHWEITE. Unter periodischen Randbedingungen ist die groesste sinnvolle
    Distanz L/2 ~ 6.4 A (Zelle ~12.9 A). Das Rezeptivfeld von ~12 A laesst sich in
    dieser Zelle also NICHT vollstaendig ausmessen. Der Test ist damit
    notwendigerweise unvollstaendig: er kann kurzreichweitige Korrelation
    bestaetigen, aber langreichweitige nicht ausschliessen.

KONTROLLE: dieselbe Rechnung mit zufaellig permutierten dF_i. Das zerstoert jede
raeumliche Struktur und liefert das Rauschniveau.

Kosten: braucht Kraefte, also einmal get_predictions (MACE ueber alle Frames).
Danach gecacht.

Ausfuehren:
    python analyses/11_error_correlation/force_error_correlation.py
    python analyses/11_error_correlation/force_error_correlation.py --ensemble ensemble_L0
    python analyses/11_error_correlation/force_error_correlation.py --member 0 --bins 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.data import TEST_SET_BIG, TEST_SET_SMALL, load_trajectory
from uq_mace.predictions import get_predictions

HERE = Path(__file__).resolve().parent
TEST_SETS = {"big": TEST_SET_BIG, "small": TEST_SET_SMALL}


# ---------------------------------------------------------------------------
def accumulate(frames, forces, f_ref, member, bins, rng, shuffle=False):
    """Bin-weise Summe von dF_i . dF_j gegen den Abstand r_ij (mit MIC)."""
    edges = bins
    n_bins = len(edges) - 1
    s_dot = np.zeros(n_bins)     # Summe der Skalarprodukte je Bin
    n_pair = np.zeros(n_bins)    # Zahl der Paare je Bin
    s_sq = 0.0                   # Summe |dF_i|^2 (Normierung)
    n_at = 0

    for atoms, f_ml, f_dft in zip(frames, forces, f_ref):
        fm = f_ml[member] if f_ml.ndim == 3 else f_ml
        dF = np.asarray(f_dft) - np.asarray(fm)          # (N, 3)
        if shuffle:
            dF = dF[rng.permutation(dF.shape[0])]        # Ortsstruktur zerstoeren

        s_sq += float((dF ** 2).sum())
        n_at += dF.shape[0]

        D = atoms.get_all_distances(mic=True)            # (N, N), min. image
        G = dF @ dF.T                                    # (N, N) Skalarprodukte
        iu = np.triu_indices(D.shape[0], k=1)            # nur i<j, ohne Diagonale
        idx = np.digitize(D[iu], edges) - 1
        ok = (idx >= 0) & (idx < n_bins)
        np.add.at(s_dot, idx[ok], G[iu][ok])
        np.add.at(n_pair, idx[ok], 1.0)

    mean_sq = s_sq / n_at                                # <|dF|^2>
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(n_pair > 0, s_dot / np.maximum(n_pair, 1) / mean_sq, np.nan)
        # grobe Fehlerbalken: Standardfehler des Mittels je Bin
        err = np.where(n_pair > 0, 1.0 / np.sqrt(np.maximum(n_pair, 1)), np.nan)
    return C, err, n_pair, mean_sq


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="ensemble_L2c")
    ap.add_argument("--testset", default="big", choices=["big", "small"])
    ap.add_argument("--member", type=int, default=0,
                    help="welches Ensemble-Member (Default 0; einzelnes Modell ist "
                         "naeher am realen Reweighting-Fall)")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--rmax", type=float, default=None,
                    help="max. Abstand [A]; Default: kleinste L/2 ueber alle Frames")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    frames = load_trajectory(TEST_SETS[args.testset])
    pred = get_predictions(args.ensemble, args.testset, device=args.device)
    forces, f_ref = pred["forces"], pred["f_ref"]

    # groesste unter MIC sinnvolle Distanz = min(L)/2
    half = min(min(np.linalg.norm(a.cell.array, axis=1)) / 2.0 for a in frames)
    rmax = args.rmax if args.rmax else half
    edges = np.linspace(0.0, rmax, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_at = len(frames[0])
    baseline = -1.0 / (n_at - 1)     # Summenregel: sum_i dF_i = 0

    print(f"[calc ] {len(frames)} Frames, Member {args.member}, "
          f"r bis {rmax:.2f} A ({args.bins} Bins)")
    C, err, npair, msq = accumulate(frames, forces, f_ref, args.member, edges, rng)
    Cs, _, _, _ = accumulate(frames, forces, f_ref, args.member, edges, rng, shuffle=True)

    print(f"[calc ] <|dF|^2>^(1/2) = {np.sqrt(msq)*1000:.1f} meV/A pro Atom")
    print(f"[calc ] Summenregel-Grundlinie: -1/(N-1) = {baseline:+.5f}\n")

    # ---- Auswertung: wo faellt C(r) auf die Grundlinie? ----
    print(f"{'r [A]':>8}{'C(r)':>10}{'±':>8}{'Paare':>12}")
    for i in range(0, args.bins, max(1, args.bins // 12)):
        if npair[i] > 0:
            print(f"{centers[i]:>8.2f}{C[i]:>10.4f}{err[i]:>8.4f}{int(npair[i]):>12d}")

    finite = np.isfinite(C)
    within = np.abs(C - baseline) <= 2 * err
    idx = np.where(finite & ~within)[0]
    if idx.size:
        r_corr = centers[idx[-1]]
        print(f"\n  Letzter Bin signifikant ueber der Grundlinie: r = {r_corr:.2f} A")
        if r_corr > 0.8 * rmax:
            print("  !! reicht bis an den Rand des messbaren Bereichs -> Korrelation")
            print("     koennte weiter reichen als die Zelle zulaesst. NICHT bestaetigt.")
        else:
            print(f"  -> Korrelationslaenge ~{r_corr:.1f} A, deutlich unter r_max."
                  f" sqrt(N)-Annahme gestuetzt.")
    else:
        print("\n  Kein Bin signifikant von der Grundlinie verschieden.")

    # ---- Plot ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    a1.errorbar(centers, C, yerr=err, fmt="o-", ms=4, lw=1.5, color="steelblue",
                label="$C(r)$ — Kraftfehler")
    a1.plot(centers, Cs, "s--", ms=3, lw=1.0, color="gray", alpha=0.7,
            label="Kontrolle (dF permutiert)")
    a1.axhline(baseline, color="crimson", ls="--", lw=1.4,
               label=f"Summenregel $-1/(N-1)$ = {baseline:.4f}")
    a1.axhline(0, color="k", lw=0.8, alpha=0.4)
    a1.set_xlabel("Abstand $r$ [Å]")
    a1.set_ylabel("$C(r) = \\langle\\delta F_i\\cdot\\delta F_j\\rangle / \\langle|\\delta F|^2\\rangle$")
    a1.set_title("(a) Räumliche Korrelation des Kraftfehlers")
    a1.legend(fontsize=8.5); a1.grid(alpha=0.3)

    # (b) Zoom auf die Grundlinie, log-Betrag der Abweichung
    dev = np.abs(C - baseline)
    a2.semilogy(centers, np.maximum(dev, 1e-6), "o-", ms=4, color="steelblue",
                label="$|C(r) - $Grundlinie$|$")
    a2.semilogy(centers, np.maximum(err, 1e-6), "--", color="crimson", lw=1.3,
                label="1σ Rauschniveau")
    a2.set_xlabel("Abstand $r$ [Å]")
    a2.set_ylabel("Abweichung von der Grundlinie")
    a2.set_title("(b) Wo verschwindet die Korrelation im Rauschen?")
    a2.legend(fontsize=8.5); a2.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Kraftfehler-Korrelation — Test der $\\sqrt{{N}}$-Annahme   |   "
        f"{args.ensemble}, Member {args.member}, test{args.testset}\n"
        f"messbar nur bis $L/2$ = {rmax:.1f} Å; MACE-Rezeptivfeld ~12 Å bleibt "
        f"damit unerreichbar", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = HERE / f"force_error_correlation_{args.ensemble}.png"
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] {out}")


if __name__ == "__main__":
    main()
