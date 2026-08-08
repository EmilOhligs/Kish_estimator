"""Guetefunktion des einseitigen FAIL-only-Monitors ueber z.

Eigenstaendige Fassung von §7.8 aus Results/UQ_L0/screening_methode.ipynb.
Im Notebook musste N_OC klein bleiben (Laufzeitbudget); hier sind die
Stichprobenzahlen oben als Knoepfe herausgezogen.

Aufruf (aus dem Projekt-Root oder von ueberall):
    python analyses/13_sequential_screening/operating_characteristic.py

Worum es geht
-------------
Die Regel feuert FAIL, wenn die untere Grenze des Konfidenzintervalls der
Entscheidungsgroesse D = c - c_max ueber null liegt. z steuert dabei den
Arbeitspunkt zwischen zwei ungleich teuren Fehlern:

  Falsch-FAIL      -> ein brauchbares Modell wird abgebrochen (das Ergebnis ist weg)
  verpasstes FAIL  -> ein untaugliches Modell laeuft durch (kostet nur Rechenzeit;
                      am Ende steht der exakte Kish-Wert und man sieht es)

Aufloesungsgrenze: bei N_OC Sequenzen ohne einen einzigen Treffer ist die Rate
nur nach oben durch etwa 3/N_OC beschraenkt (Dreierregel). Fuer eine Aussage
"unter 0.1 %" braucht es also mindestens ~3000 Sequenzen je Punkt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Parameter
# ----------------------------------------------------------------------------
R        = 0.8                                    # Effizienzziel N_eff/n >= R
Z_GRID   = (1.28, 1.64, 2.00, 2.33, 2.58, 3.00)   # zu untersuchende Arbeitspunkte
RHO_GRID = (0.90, 0.95, 1.05, 1.10, 1.20)         # c_true / c_max^schief
N_OC     = 4000        # Sequenzen je (z, rho)  -- im Notebook nur 250
K_MAX    = 400         # Laenge einer Sequenz im Grenzfall-Teil
B_BOOT   = 100         # Bootstrap-Resamples je Checkpoint
N_REAL   = 500         # Sequenzen je echtem Modell
K_REAL   = 500         # Laenge einer Sequenz im Realdaten-Teil
Z_REAL   = (1.64, 2.00, 2.58, 3.00)
SEED     = 3
MAKE_PLOT = True

CHECKPOINTS = np.array([5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300, 400, 500])
K_FLOOR = 5
T_KELVIN = 292.0
K_B = 8.617333262e-5

# ----------------------------------------------------------------------------
# Projekt-Setup
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve()
while not (ROOT / "src" / "uq_mace").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / "cache"

from uq_mace.predictions import load_energies  # noqa: E402

beta = 1.0 / (K_B * T_KELVIN)
LNR = float(np.log(R))
CMAX_GAUSS = float(np.sqrt(-LNR))

MODELS_FULL = {
    "mace-L0-01":   "single_mace-L0-01_testfull_n5000.npz",
    "mace-L0-c-01": "single_mace-L0-c-01_testfull_n5000.npz",
    "mace-L2-c-01": "single_mace-L2-c-01_testfull_n5000.npz",
}

# Die 4 fehlerhaften DFT-Frames ueber die praezise L2-Referenz identifizieren
_ed, _el2 = load_energies(CACHE / MODELS_FULL["mace-L2-c-01"])
BAD = (_ed - _el2) > 0.10


def load_full(name: str):
    """e_dft, e_mace des n5000-Satzes ohne die 4 DFT-Artefakte."""
    e_dft, e_mace = load_energies(CACHE / MODELS_FULL[name])
    return e_dft[~BAD], e_mace[~BAD]


# ----------------------------------------------------------------------------
# Schranke und Entscheidungsgroesse
# ----------------------------------------------------------------------------
def cmax_skew_vec(g1, g2, iters: int = 6):
    """c_max^schief fuer ganze Arrays -- Newton ab der Gauss-Loesung.

    Die Schranke loest c^2 - g1 c^3 + 7/12 g2 c^4 = -ln R und haengt nur von
    (g1, g2, R) ab, nicht von c. Sie liegt fuer R in [0.7, 0.9] bei c ~ 0.35-0.68,
    also im konvergenten Bereich der Kumulantenreihe -- anders als c selbst, das
    bei den L0-Modellen bis 1.4 geht, wo die Reihe wertlos ist. Deshalb wird die
    SCHRANKE ausgewertet und c mit ihr verglichen, nie die Reihe bei c.

    Konvergiert Newton nicht (Reihe hat dort keine Wurzel), Rueckfall auf Gauss.
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    c = np.full(np.broadcast(g1, g2).shape, CMAX_GAUSS, dtype=float)
    for _ in range(iters):
        f = c**2 - g1 * c**3 + (7 / 12) * g2 * c**4 + LNR
        fp = 2 * c - 3 * g1 * c**2 + (7 / 3) * g2 * c**3
        step = np.where(np.abs(fp) > 1e-12, f / fp, 0.0)
        c = np.clip(c - step, 0.05, 3.0)
    resid = c**2 - g1 * c**3 + (7 / 12) * g2 * c**4 + LNR
    return np.where(np.abs(resid) < 1e-6, c, CMAX_GAUSS)


def stat_D(X):
    """Entscheidungsgroesse D = c - c_max je Zeile von X (n, k), plus c, g1, g2.

    Vollstaendig kausal: benutzt nur die uebergebenen k Punkte.
    """
    k = X.shape[1]
    Y = X - X.mean(1, keepdims=True)
    m2 = (Y**2).mean(1)
    m3 = (Y**3).mean(1)
    m4 = (Y**4).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = beta * np.sqrt(np.maximum(m2 * k / max(k - 1, 1), 0.0))
        g1 = np.where(m2 > 0, m3 / m2**1.5, 0.0)
        g2 = np.where(m2 > 0, m4 / m2**2 - 3.0, 0.0)
    return c - cmax_skew_vec(g1, g2), c, g1, g2


def monitor_boot(seqs, z, B=B_BOOT, rng=None, chunk=250):
    """Einseitiger FAIL-Monitor: FAIL sobald D(k) > z * SE_boot(D(k)).

    Prueft nur an CHECKPOINTS, behauptet nie PASS.
    Rueckgabe: (gefeuert, k_bei_dem_gefeuert); k = -1 heisst 'nie'.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ck = CHECKPOINTS[CHECKPOINTS <= seqs.shape[1]]
    n = seqs.shape[0]
    fired = np.zeros(n, dtype=bool)
    kfire = np.full(n, -1)
    for k in ck:
        todo = np.where(~fired)[0]
        if todo.size == 0:
            break
        Dk = stat_D(seqs[todo, :k])[0]
        se = np.empty(todo.size)
        for a in range(0, todo.size, chunk):      # gechunkt, sonst Speicher
            b = min(a + chunk, todo.size)
            sub = seqs[todo[a:b], :k]
            idx = rng.integers(0, k, (b - a, B, k))
            boot = np.take_along_axis(sub[:, None, :], idx, axis=2)
            se[a:b] = stat_D(boot.reshape(-1, k))[0].reshape(b - a, B).std(1)
        sel = todo[Dk > z * se]
        fired[sel] = True
        kfire[sel] = k
    return fired, kfire


# ----------------------------------------------------------------------------
# Hilfsmittel fuer den synthetischen Grenzfall
# ----------------------------------------------------------------------------
def rescale_to(dE, c_target):
    """Skala auf c_target setzen, Form (g1, g2) unveraendert lassen."""
    mu = dE.mean()
    return mu + (c_target / (beta * dE.std(ddof=1))) * (dE - mu)


def kish_truth(pool, N=400_000, seed=99):
    """Schrankenfreie Wahrheit: exaktes Kish auf sehr grosser Stichprobe.

    Achtung: das beseitigt nur das Monte-Carlo-Rauschen. Gezogen wird weiterhin
    aus der endlichen empirischen Verteilung -- alles jenseits ihres Maximums
    fehlt per Konstruktion (Annahme A2).
    """
    d = pool[np.random.default_rng(seed).integers(0, pool.size, N)]
    w = np.exp(-beta * (d - d.mean()))
    return (w.sum() ** 2 / np.sum(w**2)) / N


def rule_of_three(n_hit, n_tot):
    """Obere 95-%-Schranke, wenn nichts (oder fast nichts) beobachtet wurde."""
    if n_hit == 0:
        return 100.0 * 3.0 / n_tot
    return None


# ----------------------------------------------------------------------------
# Teil A -- Guetefunktion ueber z auf synthetischen Grenzfaellen
# ----------------------------------------------------------------------------
def part_a():
    dE_b = np.subtract(*load_energies(CACHE / "mace_energies_ensemble_L2c_testbig.npz"))
    s = dE_b.std(ddof=1)
    u = dE_b - dE_b.mean()
    g1 = float((u**3).mean() / s**3)
    g2 = float((u**4).mean() / s**4 - 3.0)
    CS = float(cmax_skew_vec(np.array([g1]), np.array([g2]))[0])

    print("=" * 88)
    print("Teil A -- Guetefunktion ueber z (synthetische Grenzfaelle)")
    print("=" * 88)
    print(f"Basis: ensemble_L2c, gamma1={g1:+.3f}, gamma2={g2:+.3f}")
    print(f"c_max^Gauss={CMAX_GAUSS:.4f}   c_max^schief={CS:.4f}   R={R}")
    print(f"{N_OC} Sequenzen a {K_MAX} Punkte je (z, rho), B={B_BOOT}")
    print(f"Aufloesungsgrenze bei 0 Treffern: <= {100*3/N_OC:.2f} % (Dreierregel)\n")

    # Sequenzen einmal ziehen und fuer ALLE z wiederverwenden -> fairer Vergleich
    rng0 = np.random.default_rng(SEED)
    seqs, truth = {}, {}
    for rho in RHO_GRID:
        pool = rescale_to(dE_b, rho * CS)
        seqs[rho] = pool[rng0.integers(0, pool.size, (N_OC, K_MAX))]
        truth[rho] = kish_truth(pool)

    print("Wahres N_eff/n je rho (exaktes Kish, schrankenfrei):")
    for rho in RHO_GRID:
        print(f"   rho={rho:.2f}  ->  N_eff/n = {truth[rho]:.4f}  "
              f"({'PASS' if truth[rho] >= R else 'FAIL'})")
    print()

    head = f"{'z':>6}{'nominal':>10}"
    for rho in RHO_GRID:
        head += f"{'rho='+format(rho, '.2f'):>11}"
    print(head)
    print(f"{'':>6}{'':>10}" + "".join(
        f"{('Fehler' if truth[r] >= R else 'Erkenn'):>11}" for r in RHO_GRID))
    print("-" * len(head))

    from scipy.stats import norm
    table = {}
    for z in Z_GRID:
        row = []
        for rho in RHO_GRID:
            fired, _ = monitor_boot(seqs[rho], z=z, rng=np.random.default_rng(SEED))
            row.append(100.0 * fired.mean())
        table[z] = row
        line = f"{z:>6.2f}{100*(1-norm.cdf(z)):>9.1f}%"
        for rho, v in zip(RHO_GRID, row):
            bound = rule_of_three(int(round(v * N_OC / 100)), N_OC)
            line += f"{('<'+format(bound,'.2f') if (v == 0 and bound) else format(v,'.2f')):>10}%"
        print(line)

    print("\n'Fehler' = Falsch-FAIL (wahr PASS)  |  'Erkenn' = korrekt erkannt (wahr FAIL)")
    return table, CS


# ----------------------------------------------------------------------------
# Teil B -- was ein groesseres z bei den echten Modellen kostet
# ----------------------------------------------------------------------------
def part_b():
    print("\n" + "=" * 88)
    print("Teil B -- dieselbe Frage auf den echten Modellen")
    print("=" * 88)
    print(f"{N_REAL} Sequenzen a {K_REAL} Punkte je Modell, B={B_BOOT}\n")

    head = f"{'Modell':<15}{'rho':>6}{'N_eff/n':>9}"
    for z in Z_REAL:
        head += f"{'z='+format(z,'.2f'):>17}"
    print(head)
    print(f"{'':<15}{'':>6}{'':>9}" + "".join(f"{'gefeuert':>10}{'k_med':>7}"
                                              for _ in Z_REAL))
    print("-" * len(head))

    for model in MODELS_FULL:
        e_dft, e_mace = load_full(model)
        rng = np.random.default_rng(11)
        idx = rng.integers(0, e_dft.size, (N_REAL, K_REAL))
        seqs = (e_dft - e_mace)[idx]

        _, c_m, g1_m, g2_m = stat_D(seqs)
        rho_m = float(np.median(c_m / cmax_skew_vec(g1_m, g2_m)))
        w = np.exp(-beta * (seqs - seqs.min(axis=1, keepdims=True)))
        neff = float(np.median((w.sum(1) ** 2 / np.sum(w**2, 1)) / K_REAL))

        line = f"{model:<15}{rho_m:>6.2f}{neff:>9.3f}"
        for z in Z_REAL:
            fired, kfire = monitor_boot(seqs, z=z, rng=np.random.default_rng(7))
            km = np.median(kfire[fired]) if fired.any() else np.nan
            line += f"{fired.sum():>6}/{N_REAL:<3}" + (
                f"{km:>7.0f}" if fired.any() else f"{'--':>7}")
        print(line)

    print("\nrho und N_eff/n sind Mediane ueber die Sequenzen.")


# ----------------------------------------------------------------------------
def make_plot(table, CS):
    import matplotlib.pyplot as plt

    zz = np.array(Z_GRID)
    pass_rhos = [r for r in RHO_GRID if r < 1.0]
    fail_rhos = [r for r in RHO_GRID if r > 1.0]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8))
    for r in pass_rhos:
        j = RHO_GRID.index(r)
        ax.plot(zz, [table[z][j] for z in Z_GRID], "o-", lw=2,
                label=fr"Fehlalarm $\rho$={r}")
    for r in fail_rhos:
        j = RHO_GRID.index(r)
        ax.plot(zz, [table[z][j] for z in Z_GRID], "s--", lw=1.8,
                label=fr"Erkennung $\rho$={r}")
    ax.set_xlabel("$z$")
    ax.set_ylabel("FAIL gefeuert [%]")
    ax.set_ylim(-3, 105)
    ax.set_title("Beide Fehlerarten laufen gegenläufig")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    jx = RHO_GRID.index(0.95)
    jy = RHO_GRID.index(1.10)
    bx.plot([table[z][jx] for z in Z_GRID], [table[z][jy] for z in Z_GRID],
            "o-", color="steelblue", lw=2, ms=7)
    for z in Z_GRID:
        bx.annotate(f"z={z:.2f}", (table[z][jx], table[z][jy]),
                    textcoords="offset points", xytext=(7, -3), fontsize=8)
    bx.set_xlabel(r"Fehlalarm bei $\rho=0.95$ [%]   (teuer: Ergebnis weg)")
    bx.set_ylabel(r"Erkennung bei $\rho=1.10$ [%]   (billig: nur Rechenzeit)")
    bx.set_title("Der Tauschkurs — die Achsen kosten nicht dasselbe")
    bx.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).with_name("operating_characteristic.png")
    plt.savefig(out, dpi=130)
    print(f"\nAbbildung geschrieben: {out}")


if __name__ == "__main__":
    table, CS = part_a()
    part_b()
    if MAKE_PLOT:
        make_plot(table, CS)
