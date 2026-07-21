"""
Konvergenz von c und gamma_1: intrinsisch vs. Pool-Artefakt
============================================================================

Umsetzung von notebooks/plan_konvergenz_simulation.md.

FRAGE: Bei n = 400 stabilisiert sich die laufende Schiefe scheinbar ab k ~ 210.
Ist das die echte statistische Konvergenz - oder ein Artefakt des kleinen Pools?

HYPOTHESE: Das Permutationsband misst nicht die Stichprobenunsicherheit, sondern
nur die Reihenfolgeabhaengigkeit innerhalb eines FESTEN Pools. Beim Ziehen ohne
Zuruecklegen gilt die Endlichkeitskorrektur

    U_perm(k; n)  ~  U_iid(k) * sqrt((n-k)/(n-1))

die das Band bei k = n zwangsweise auf null druecken muss - unabhaengig davon,
ob die Groesse tatsaechlich konvergiert ist.

EXPERIMENTE
  0  Generator gegen die realen Daten validieren
  A  U_iid(k)   - unabhaengige Ziehungen, die Referenz (haengt nur von k)
  B  U_perm(k;n)- Permutationsband fuer mehrere Poolgroessen; Test der Hypothese
  C  k*(n)      - Konvergenzpunkt aus Band vs. aus der Wahrheit
  D  U_boot(k;n)- Bootstrap als ehrliche Alternative
  E  optional   - dasselbe fuer k_hat  (--with-khat, teuer)

Ausfuehren:
    python analyses/convergence_simulation.py
    python analyses/convergence_simulation.py --reps 1000 --pools 20 --with-khat
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.distributions import StdSkewNormal
from uq_mace.reweighting import psis_khat, running_moments

HERE = Path(__file__).resolve().parent   # Ausgaben landen neben dem Skript
K_B = 8.617333262e-5  # eV/K

# Messwerte aus cache/mace_energies_ensemble_L2c_testbig.npz (L2c, 400 Frames, 300 K)
REAL_C = 0.3180
REAL_SKEW = 0.5027
REAL_KURT = 0.4071
REAL_N = 400


# ---------------------------------------------------------------------------
# Momente einer (R, k)-Matrix entlang der Zeilen - vektorisiert
# ---------------------------------------------------------------------------
def batch_moments(z: np.ndarray):
    """Gibt (std_ddof1, skew) je Zeile zurueck."""
    k = z.shape[1]
    mu = z.mean(axis=1, keepdims=True)
    d = z - mu
    m2 = (d ** 2).mean(axis=1)
    m3 = (d ** 3).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(m2 * k / (k - 1))
        skew = np.where(m2 > 0, m3 / np.power(m2, 1.5), np.nan)
    return std, skew


def spread(a: np.ndarray) -> float:
    """Bandbreite als 10-90-%-Abstand (robust gegen Ausreisser)."""
    lo, hi = np.nanpercentile(a, [10, 90])
    return float(hi - lo)


# ---------------------------------------------------------------------------
# Experiment 0 - Generator validieren
# ---------------------------------------------------------------------------
def experiment_0(dist: StdSkewNormal, rng, reps: int = 400):
    print("=" * 78)
    print("EXPERIMENT 0 - Generator gegen reale Daten")
    print("=" * 78)
    kurt_gen = dist.excess_kurtosis()
    print(f"  Ziel-Schiefe   {REAL_SKEW:+.4f}  ->  Generator a = {dist.a:.4f}")
    print(f"  Kurtosis: real {REAL_KURT:+.4f}  vs. Generator {kurt_gen:+.4f}  "
          f"(Differenz {kurt_gen - REAL_KURT:+.4f})")
    if abs(kurt_gen - REAL_KURT) > 0.2:
        print("  !! Die Skew-Normal koppelt Schiefe und Kurtosis; die reale")
        print("     Kombination wird nicht gut getroffen -> Johnson-SU erwaegen.")
    else:
        print("  -> akzeptabel: die Familie trifft beide Momente hinreichend.")

    z = dist.rvs((reps, REAL_N), rng)
    std, sk = batch_moments(z)
    c = REAL_C * std
    print(f"\n  Ziehungen der Groesse {REAL_N} (x{reps}):")
    print(f"    c        Median {np.median(c):.4f}   10-90 % "
          f"[{np.percentile(c,10):.4f}, {np.percentile(c,90):.4f}]   real {REAL_C:.4f}")
    print(f"    gamma_1  Median {np.median(sk):+.4f}   10-90 % "
          f"[{np.percentile(sk,10):+.4f}, {np.percentile(sk,90):+.4f}]   real {REAL_SKEW:+.4f}")

    err_crit = abs(REAL_SKEW) * REAL_C ** 3
    err_exact = abs(dist.gauss_predictor_error(REAL_C))
    print(f"\n  Kriterium |gamma_1|c^3 = {err_crit*100:.3f} %")
    print(f"  exakt (via MGF)        = {err_exact*100:.3f} %   <- Guete der Naeherung")
    return err_crit


# ---------------------------------------------------------------------------
# Experiment A - intrinsische Unsicherheit U_iid(k)
# ---------------------------------------------------------------------------
def experiment_A(dist, ks, reps, rng):
    print("\n" + "=" * 78)
    print(f"EXPERIMENT A - U_iid(k), {reps} unabhaengige Ziehungen je k")
    print("=" * 78)
    out = {}
    print(f"  {'k':>6} {'med(gamma1)':>12} {'U_iid(g1)':>11} {'sqrt(6/k)*2.56':>15} "
          f"{'U_iid(c)':>10} {'U_iid(err)':>11}")
    for k in ks:
        z = dist.rvs((reps, int(k)), rng)
        std, sk = batch_moments(z)
        c = REAL_C * std
        err = np.abs(sk) * c ** 3
        out[int(k)] = dict(g1_med=float(np.nanmedian(sk)), g1_spread=spread(sk),
                           c_med=float(np.nanmedian(c)), c_spread=spread(c),
                           err_med=float(np.nanmedian(err)), err_spread=spread(err))
        # 10-90 % entspricht bei Normalitaet 2.563 Standardabweichungen
        print(f"  {k:>6} {out[int(k)]['g1_med']:>12.4f} {out[int(k)]['g1_spread']:>11.4f} "
              f"{2.563*np.sqrt(6.0/k):>15.4f} {out[int(k)]['c_spread']:>10.4f} "
              f"{out[int(k)]['err_spread']:>11.5f}")
    print("\n  Hinweis: sqrt(6/k) gilt streng nur fuer NORMALverteilte Daten;")
    print("  bei Schiefe ist die wahre Streuung groesser. Spalte dient dem Vergleich.")
    return out


# ---------------------------------------------------------------------------
# Experiment B - Permutationsband U_perm(k; n)
# ---------------------------------------------------------------------------
def experiment_B(dist, ns, ks, pools, perms, rng):
    print("\n" + "=" * 78)
    print(f"EXPERIMENT B - U_perm(k;n), {pools} Pools x {perms} Permutationen")
    print("=" * 78)
    out = {}
    for n in ns:
        kk = [int(k) for k in ks if k <= n]
        g1_sp = np.zeros((pools, len(kk)))
        c_sp = np.zeros((pools, len(kk)))
        err_sp = np.zeros((pools, len(kk)))
        for p in range(pools):
            pool = dist.rvs(int(n), rng)
            g1_curves = np.empty((perms, len(kk)))
            c_curves = np.empty((perms, len(kk)))
            for s in range(perms):
                perm = pool if s == 0 else rng.permutation(pool)
                mom = running_moments(perm)
                idx = np.array(kk) - 1
                g1_curves[s] = mom["skew"][idx]
                c_curves[s] = REAL_C * mom["std"][idx]
            e_curves = np.abs(g1_curves) * c_curves ** 3
            g1_sp[p] = [spread(g1_curves[:, j]) for j in range(len(kk))]
            c_sp[p] = [spread(c_curves[:, j]) for j in range(len(kk))]
            err_sp[p] = [spread(e_curves[:, j]) for j in range(len(kk))]
        out[int(n)] = dict(ks=np.array(kk), g1=g1_sp.mean(0),
                           c=c_sp.mean(0), err=err_sp.mean(0))
        print(f"  n = {n:>6}:  U_perm(gamma1) bei k=min .. k=n:  "
              f"{out[int(n)]['g1'][0]:.4f} .. {out[int(n)]['g1'][-1]:.4f}"
              f"   (muss am Ende -> 0 gehen)")
    return out


# ---------------------------------------------------------------------------
# Experiment D - Bootstrap
# ---------------------------------------------------------------------------
def experiment_D(dist, ns, ks, pools, perms, rng):
    print("\n" + "=" * 78)
    print("EXPERIMENT D - Bootstrap (mit Zuruecklegen)")
    print("=" * 78)
    out = {}
    for n in ns:
        kk = [int(k) for k in ks if k <= n]
        g1_sp = np.zeros((pools, len(kk)))
        for p in range(pools):
            pool = dist.rvs(int(n), rng)
            for j, k in enumerate(kk):
                idx = rng.integers(0, n, size=(perms, k))
                _, sk = batch_moments(pool[idx])
                g1_sp[p, j] = spread(sk)
        out[int(n)] = dict(ks=np.array(kk), g1=g1_sp.mean(0))
        print(f"  n = {n:>6}:  U_boot(gamma1) bei k=n = {out[int(n)]['g1'][-1]:.4f}"
              f"   (sollte NICHT auf 0 gehen)")
    return out


# ---------------------------------------------------------------------------
# Experiment C - Konvergenzpunkt k*
# ---------------------------------------------------------------------------
def k_star(ks, spreads, med, truth, tol):
    """Kleinstes k, ab dem das halbe Band innerhalb von tol*truth liegt."""
    ks = np.asarray(ks); spreads = np.asarray(spreads)
    ok = (spreads / 2.0) <= tol * abs(truth)
    bad = np.where(~ok)[0]
    if bad.size == 0:
        return int(ks[0])
    if bad[-1] + 1 >= ks.size:
        return None  # innerhalb des Gitters nie erreicht
    return int(ks[bad[-1] + 1])


def experiment_C(A, B, err_true, tols=(0.10, 0.25, 0.50)):
    print("\n" + "=" * 78)
    print("EXPERIMENT C - Konvergenzpunkt k* fuer |gamma_1|c^3")
    print("=" * 78)
    ks_A = np.array(sorted(A))
    err_sp_A = np.array([A[k]["err_spread"] for k in ks_A])
    err_md_A = np.array([A[k]["err_med"] for k in ks_A])

    rows = []
    for tol in tols:
        k_iid = k_star(ks_A, err_sp_A, err_md_A, err_true, tol)
        print(f"\n  Toleranz +-{tol*100:.0f} %   (Wahrheit = {err_true*100:.3f} %)")
        print(f"    {'Quelle':<22}{'k*':>8}{'Verzerrung':>14}")
        print(f"    {'iid (Wahrheit)':<22}{str(k_iid):>8}{'-':>14}")
        for n in sorted(B):
            ks_B = B[n]["ks"]
            k_perm = k_star(ks_B, B[n]["err"], None, err_true, tol)
            fac = (f"{k_iid/k_perm:.2f}x zu klein"
                   if (k_perm and k_iid) else "n/a")
            print(f"    {'Permutation n=' + str(n):<22}{str(k_perm):>8}{fac:>14}")
            rows.append(dict(tol=tol, n=n, k_perm=k_perm, k_iid=k_iid))
    return rows, ks_A, err_sp_A


# ---------------------------------------------------------------------------
# Experiment E - k_hat (optional)
# ---------------------------------------------------------------------------
def experiment_E(dist, ns, ks, reps, perms, rng):
    print("\n" + "=" * 78)
    print("EXPERIMENT E - dasselbe fuer k_hat")
    print("=" * 78)
    ks_e = [int(k) for k in ks if k >= 50]
    iid = []
    for k in ks_e:
        vals = [psis_khat(np.exp(-REAL_C * dist.rvs(k, rng))) for _ in range(reps)]
        iid.append(spread(np.array(vals)))
        print(f"  k={k:>6}  U_iid(khat) = {iid[-1]:.4f}")
    perm = {}
    for n in ns:
        kk = [k for k in ks_e if k <= n]
        pool_w = np.exp(-REAL_C * dist.rvs(int(n), rng))
        sp = []
        for k in kk:
            vals = [psis_khat(rng.permutation(pool_w)[:k]) for _ in range(perms)]
            sp.append(spread(np.array(vals)))
        perm[int(n)] = dict(ks=np.array(kk), khat=np.array(sp))
        print(f"  n={n:>6}  U_perm(khat) bei k=n = {sp[-1]:.4f}")
    return np.array(ks_e), np.array(iid), perm


# ---------------------------------------------------------------------------
def make_plots(A, B, D, E, err_true, outdir, ns):
    ks_A = np.array(sorted(A))
    g1_iid = np.array([A[k]["g1_spread"] for k in ks_A])
    err_iid = np.array([A[k]["err_spread"] for k in ks_A])
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ns)))

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    (a1, a2), (a3, a4) = ax

    # (a) Bandbreite gamma_1 vs k
    a1.loglog(ks_A, g1_iid, "k-", lw=2.2, label="$U_\\mathrm{iid}$ (Wahrheit)")
    for col, n in zip(colors, ns):
        a1.loglog(B[n]["ks"], B[n]["g1"], "o--", ms=4, color=col,
                  label=f"Permutation, n={n}")
    a1.set_xlabel("k"); a1.set_ylabel("Bandbreite von $\\gamma_1$ (10-90 %)")
    a1.set_title("(a) Permutationsband kollabiert bei k=n")
    a1.grid(alpha=0.3, which="both"); a1.legend(fontsize=8)

    # (b) Test der FPC-Hypothese
    for col, n in zip(colors, ns):
        ratio = B[n]["g1"] / np.interp(B[n]["ks"], ks_A, g1_iid)
        a2.plot(B[n]["ks"] / n, ratio, "o", ms=5, color=col, label=f"n={n}")
    x = np.linspace(0, 1, 200)
    a2.plot(x, np.sqrt(1 - x), "r-", lw=2, label="$\\sqrt{1-k/n}$ (Vorhersage)")
    a2.set_xlabel("k / n"); a2.set_ylabel("$U_\\mathrm{perm} / U_\\mathrm{iid}$")
    a2.set_title("(b) Test: fallen alle n auf eine Kurve?")
    a2.grid(alpha=0.3); a2.legend(fontsize=8); a2.set_ylim(0, 1.3)

    # (c) Bootstrap-Vergleich
    a3.loglog(ks_A, g1_iid, "k-", lw=2.2, label="$U_\\mathrm{iid}$")
    for col, n in zip(colors, ns):
        if n in D:
            a3.loglog(D[n]["ks"], D[n]["g1"], "s--", ms=4, color=col,
                      label=f"Bootstrap, n={n}")
    a3.set_xlabel("k"); a3.set_ylabel("Bandbreite von $\\gamma_1$")
    a3.set_title("(c) Bootstrap kollabiert nicht")
    a3.grid(alpha=0.3, which="both"); a3.legend(fontsize=8)

    # (d) das Kriterium selbst
    a4.loglog(ks_A, err_iid * 100, "k-", lw=2.2, label="$U_\\mathrm{iid}$")
    for col, n in zip(colors, ns):
        a4.loglog(B[n]["ks"], B[n]["err"] * 100, "o--", ms=4, color=col, label=f"n={n}")
    for tol, ls in ((0.25, ":"), (0.50, "-.")):
        a4.axhline(2 * tol * err_true * 100, color="crimson", ls=ls, lw=1.1,
                   label=f"Band = +-{tol*100:.0f} % der Wahrheit")
    a4.set_xlabel("k"); a4.set_ylabel("Bandbreite von $|\\gamma_1|c^3$  [%]")
    a4.set_title("(d) Unsicherheit des Gauss-Kriteriums")
    a4.grid(alpha=0.3, which="both"); a4.legend(fontsize=7.5)

    fig.suptitle("Konvergenz von $c$ und $\\gamma_1$: intrinsisch vs. Pool-Artefakt\n"
                 f"Generator: Skew-Normal, $\\gamma_1$={REAL_SKEW:.3f}, c={REAL_C:.3f} "
                 f"(kalibriert auf L2c/testbig/300 K)", fontsize=12)
    fig.subplots_adjust(top=0.88, hspace=0.28, wspace=0.22,
                        left=0.07, right=0.97, bottom=0.07)
    fig.savefig(outdir / "convergence_pool_effect.png", dpi=140)
    plt.close(fig)
    print(f"\n[plot ] {outdir/'convergence_pool_effect.png'}")

    if E is not None:
        ks_e, iid_e, perm_e = E
        fig2, ax2 = plt.subplots(figsize=(8, 5.5))
        ax2.loglog(ks_e, iid_e, "k-", lw=2.2, label="$U_\\mathrm{iid}(\\hat{k})$")
        for col, n in zip(colors, ns):
            if n in perm_e:
                ax2.loglog(perm_e[n]["ks"], perm_e[n]["khat"], "o--", ms=4,
                           color=col, label=f"Permutation, n={n}")
        ax2.set_xlabel("k"); ax2.set_ylabel("Bandbreite von $\\hat{k}$ (10-90 %)")
        ax2.set_title("$\\hat{k}$: Permutation unterschaetzt die Unsicherheit staerker,\n"
                      "weil sie keine neuen Extremwerte erzeugen kann")
        ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(outdir / "convergence_khat.png", dpi=140)
        plt.close(fig2)
        print(f"[plot ] {outdir/'convergence_khat.png'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=500, help="Ziehungen je k (Exp. A)")
    ap.add_argument("--pools", type=int, default=8, help="Pools je n (Exp. B/D)")
    ap.add_argument("--perms", type=int, default=80, help="Permutationen je Pool")
    ap.add_argument("--ns", default="400,1000,5000,20000")
    ap.add_argument("--kmax", type=int, default=20000)
    ap.add_argument("--kpoints", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-khat", action="store_true", help="Experiment E (teuer)")
    ap.add_argument("--outdir", default=None, help="Ausgabeordner (Default: neben dem Skript)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ns = [int(s) for s in args.ns.split(",")]
    ks = np.unique(np.round(np.geomspace(20, args.kmax, args.kpoints)).astype(int))

    dist = StdSkewNormal(REAL_SKEW)
    outdir = Path(args.outdir) if args.outdir else HERE
    outdir.mkdir(parents=True, exist_ok=True)

    err_true = experiment_0(dist, rng)
    A = experiment_A(dist, ks, args.reps, rng)
    B = experiment_B(dist, ns, ks, args.pools, args.perms, rng)
    rows, ks_A, err_sp_A = experiment_C(A, B, err_true)
    D = experiment_D(dist, ns, ks, max(2, args.pools // 2), args.perms, rng)
    E = (experiment_E(dist, ns, ks, max(50, args.reps // 5), args.perms, rng)
         if args.with_khat else None)

    make_plots(A, B, D, E, err_true, outdir, ns)

    with open(outdir / "k_star.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["tol", "n", "k_perm", "k_iid"])
        wtr.writeheader(); wtr.writerows(rows)
    print(f"[csv  ] {outdir/'k_star.csv'}")


if __name__ == "__main__":
    main()
