#!/usr/bin/env python3
"""
kish_screening.py — is reweighting worth it?
====================================================

Decides, for a given set of DFT and ML energies, whether thermodynamic
reweighting will statistically hold up, and simulates the sequential
monitor: after how many points can the run be stopped?

    python3 kish_screening.py DFT_FILE ML_FILE [options]

Exit codes (for use in shell scripts):

    0   PASS  — criterion met, reweighting holds (in live mode also:
        CONTINUE, no FAIL so far, keep the campaign running)
    1   FAIL  — stopping condition met, halt the calculation
    2   usage error (arguments, missing file, unknown format)
    3   data error (mismatched length, too few points, NaN)
    4   result not reliable (moment condition violated or series
        outside its validity range) — neither PASS nor FAIL

Example (batch, finished dataset):

    Two-file input:

    python3 kish_screening.py e_dft.npy e_mace.npy -R 0.8 || {
        echo "Reweighting does not hold — abort MD" >&2
        exit 1
    }

    Single-file input:
    python3 kish_screening.py cache/single_mace-L0-01_testfull_n5000.npz \
        -R 0.8 -T 292 --steps
    echo "Exit code: $?"

Example (live, embedded in a running MD/DFT campaign planned for 5000
points): call again for each newly arrived DFT batch, with the points
collected so far. A single call checks ONLY the checkpoint currently due
(no grid walk) and returns CONTINUE/FAIL; once all 5000 points are
available, the same call automatically delivers the full certification
(khat, exact N_eff/n, PASS/FAIL/UNCLEAR):

    python3 kish_screening.py e_dft_so_far.npy e_mace_so_far.npy \
        -R 0.8 -T 292 -N 5000 --live -q || {
        echo "FAIL — abort MD/DFT campaign" >&2
        exit 1
    }

Requires only numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

__version__ = "1.1"

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
KB_EV = 8.617333262e-5      # Boltzmann constant in eV/K
R5_TOL = 0.05               # allowed remainder (2c)^5/5! of the cumulant series
C_VALID = 0.5 * (120.0 * R5_TOL) ** 0.2      # ~0.7155, beyond this the series is worthless
KHAT_GATE = 0.5             # E[w^2] < infinity  <=>  khat < 0.5

K_FLOOR = 50                # earliest evaluation; below this not only is SE(c)
                            # large, but the BOUND itself is estimated
                            # unreliably (at k=5 the quartic has no root at
                            # all within the validity range in 43% of draws)
FIRST_FRAC = 0.10           # grid start as a fraction of n -- NOT identical
                            # to K_FLOOR, see checkpoint_grid

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_DATA, EXIT_UNRELIABLE = 0, 1, 2, 3, 4

UNITS = {"eV": 1.0, "meV": 1e-3, "Ha": 27.211386245988,
         "Ry": 13.605693122994, "kcal/mol": 0.0433641153087705,
         "kJ/mol": 0.010364269656262}


class Abbruch(Exception):
    """Controlled abort with an exit code."""

    def __init__(self, code: int, text: str):
        super().__init__(text)
        self.code = code


# --------------------------------------------------------------------------
# Reading input
# --------------------------------------------------------------------------
def lade_energien(pfad: str, key: str | None = None) -> np.ndarray:
    """Read energies from .npz, .npy, .txt/.dat/.csv. Returns: 1D float array.

    npz: uses `key` if given. Otherwise the first matching key from a
    preference list is searched for; if the match has two axes (committee),
    it is averaged over the first (member) axis.
    """
    p = Path(pfad)
    if not p.exists():
        raise Abbruch(EXIT_USAGE, f"file not found: {p}")
    if p.is_dir():
        raise Abbruch(EXIT_USAGE, f"is a directory, not a file: {p}")

    endung = p.suffix.lower()
    try:
        if endung == ".npz":
            d = np.load(p, allow_pickle=False)
            if key is not None:
                if key not in d.files:
                    raise Abbruch(EXIT_USAGE,
                                  f"{p.name}: key '{key}' missing. "
                                  f"Available: {', '.join(d.files)}")
                a = d[key]
            else:
                bevorzugt = ["e_dft", "e_mace", "e_model", "energies",
                             "energy", "E", "e"]
                treffer = next((k for k in bevorzugt if k in d.files), None)
                if treffer is None:
                    if len(d.files) == 1:
                        treffer = d.files[0]
                    else:
                        raise Abbruch(EXIT_USAGE,
                                      f"{p.name}: no unambiguous energy key. "
                                      f"Available: {', '.join(d.files)}. "
                                      f"Specify with --key-dft / --key-ml.")
                a = d[treffer]
        elif endung == ".npy":
            a = np.load(p, allow_pickle=False)
        elif endung in (".txt", ".dat", ".csv", ".tsv", ""):
            trenn = "," if endung == ".csv" else None
            a = np.loadtxt(p, delimiter=trenn, comments=("#", "!", "%"))
        else:
            raise Abbruch(EXIT_USAGE,
                          f"{p.name}: unknown extension '{endung}'. "
                          f"Supported: .npz .npy .txt .dat .csv .tsv")
    except Abbruch:
        raise
    except Exception as e:                                   # noqa: BLE001
        raise Abbruch(EXIT_USAGE, f"{p.name}: unreadable ({type(e).__name__}: {e})")

    a = np.asarray(a, dtype=float)
    if a.ndim == 2:
        # (member, frame) -> ensemble mean; (frame, 1) -> flatten
        a = a.ravel() if 1 in a.shape else a.mean(axis=0)
    if a.ndim != 1:
        raise Abbruch(EXIT_DATA, f"{p.name}: expected 1D or 2D, got {a.ndim}D {a.shape}")
    return a


def lade_paar(pfad: str) -> tuple[np.ndarray, np.ndarray]:
    """Read (e_dft, e_ml) from a SINGLE npz cache.

    Mirrors uq_mace.predictions.load_energies: requires 'e_dft' and uses
    'e_mace', otherwise the mean of 'energies' over axis 0 (member axis,
    shape (M, F)). This lets the script run directly on the project caches
    predictions_*.npz and mace_energies_*.npz.
    """
    p = Path(pfad)
    if not p.exists():
        raise Abbruch(EXIT_USAGE, f"file not found: {p}")
    if p.suffix.lower() != ".npz":
        raise Abbruch(EXIT_USAGE,
                      f"{p.name}: single-file mode needs a .npz with "
                      f"'e_dft' and 'e_mace'/'energies'. Otherwise provide "
                      f"two files.")
    try:
        d = np.load(p, allow_pickle=False)
    except Exception as e:                                   # noqa: BLE001
        raise Abbruch(EXIT_USAGE, f"{p.name}: unreadable ({type(e).__name__}: {e})")
    if "e_dft" not in d.files:
        raise Abbruch(EXIT_USAGE,
                      f"{p.name}: does not contain 'e_dft' (keys: {', '.join(d.files)})")
    e_dft = np.asarray(d["e_dft"], dtype=float)
    if "e_mace" in d.files:
        e_ml = np.asarray(d["e_mace"], dtype=float)
    elif "energies" in d.files:
        e_ml = np.asarray(d["energies"], dtype=float).mean(axis=0)
    else:
        raise Abbruch(EXIT_USAGE,
                      f"{p.name}: 'e_dft' without 'e_mace'/'energies'")
    return e_dft.ravel(), e_ml.ravel()


def pruefe_daten(e_dft: np.ndarray, e_ml: np.ndarray, k_floor: int,
                 n_plan: int | None = None) -> np.ndarray:
    """Check both arrays for compatibility, return dE.

    n_plan: planned campaign budget (live mode). Checkpoint existence is
    then checked against n_plan, not against the currently available
    e_dft.size — otherwise the check spuriously fails with few live points
    so far, even though the campaign plans enough points overall.
    """
    if e_dft.size != e_ml.size:
        raise Abbruch(EXIT_DATA,
                      f"mismatched length: DFT has {e_dft.size}, ML has {e_ml.size} values. "
                      f"Both files must describe the same frames in the same "
                      f"order.")
    if e_dft.size == 0:
        raise Abbruch(EXIT_DATA, "empty input")
    dE = e_dft - e_ml
    schlecht = ~np.isfinite(dE)
    if schlecht.any():
        raise Abbruch(EXIT_DATA,
                      f"{schlecht.sum()} non-finite values in dE "
                      f"(positions {np.where(schlecht)[0][:5].tolist()}...)")
    grenze = dE.size if n_plan is None else n_plan
    ck = checkpoints_fuer(grenze, k_floor)
    if not ck:
        raise Abbruch(EXIT_DATA,
                      f"only {grenze} points (planned) — the grid has no "
                      f"checkpoint at k >= {k_floor}. Plan more points or "
                      f"lower --k-floor (reference value {K_FLOOR}).")
    if np.std(dE) == 0.0:
        raise Abbruch(EXIT_DATA, "dE is constant — are the models identical?")
    return dE


# --------------------------------------------------------------------------
# Core quantities
# --------------------------------------------------------------------------
def momente(dE: np.ndarray, beta: float):
    """(c, gamma1, gamma2)."""
    n = dE.size
    u = dE - dE.mean()
    m2 = (u ** 2).mean()
    c = beta * np.sqrt(m2 * n / (n - 1)) if n > 1 else np.nan
    g1 = float((u ** 3).mean() / m2 ** 1.5)
    g2 = float((u ** 4).mean() / m2 ** 2 - 3.0)
    return float(c), g1, g2


def se_c(c: float, g2: float, k: int) -> float:
    """SE(c) = c * sqrt((gamma2 + 2) / 4k), delta method for the
    sample standard deviation."""
    return float(c * np.sqrt(max(g2 + 2.0, 0.0) / (4.0 * max(k, 2))))


def cmax_gauss(R: float) -> float:
    """Bound without shape correction: N_eff/n = exp(-c^2) >= R."""
    return float(np.sqrt(-np.log(R)))


def cmax_skew(R: float, g1: float, g2: float, c_hi: float | None = None,
              warn: bool = True) -> float:
    """Smallest positive root of c^2 - g1 c^3 + 7/12 g2 c^4 = -ln R,
    capped at c_hi.

    The quartic is fully factorized via the companion matrix (numpy.roots)
    instead of a bracketing method: for gamma2 < 0 there are two positive
    roots, and a bracket returns the wrong one depending on the interval.
    If no root exists within the validity range, c_hi is returned — the
    UPPER bound, so the monitor doesn't fire too early.
    """
    c_hi = C_VALID if c_hi is None else c_hi
    r = np.roots([(7.0 / 12.0) * g2, -g1, 1.0, 0.0, np.log(R)])
    pos = [x.real for x in r
           if abs(x.imag) <= 1e-8 * max(1.0, abs(x.real)) and 1e-12 < x.real < c_hi]
    if not pos:
        if warn:
            print(f"    [Note] cmax_skew: no root below C_VALID for "
                  f"g1={g1:+.3f}, g2={g2:+.3f} -> conservative fallback {c_hi:.3f}")
        return float(c_hi)
    return float(min(pos))


def log_neff_ratio(c: float, g1: float = 0.0, g2: float = 0.0) -> float:
    """log(N_eff/n) via the cumulant expansion: -c^2 + g1 c^3 - 7/12 g2 c^4.

    g1 = g2 = 0 gives the pure Gaussian predictor. The comparison with the
    exact Kish value is the cross-check: for ensemble_L2c the expansion
    explains the observed deviation from -1.50% with -1.48% almost
    completely.
    """
    return float(-c ** 2 + g1 * c ** 3 - (7.0 / 12.0) * g2 * c ** 4)


def diagnose(R: float, g1: float, g2: float, rem_tol: float = R5_TOL) -> list[str]:
    """Check the quartic's preconditions -- once per model, not per call.

    Empty list = everything is fine. Checked:

    * **Uniqueness.** f'(c) = c [(7/3) g2 c^2 - 3 g1 c + 2]; the bracket has
      no positive root when g1 <= 0 <= g2 or g1^2 < (56/27) g2. Then f is
      strictly monotonic on (0, inf) and the root is unique. Otherwise there
      may be more -- cmax_skew takes the smallest, which is correct, but the
      situation should be visible.
    * **N_eff <= n.** Cauchy-Schwarz forces log(N_eff/n) <= 0. This is
      checked directly at c_max via log_neff_ratio -- not via the roots of
      its derivative, because for g2 > 0 the invalid zone of those roots is
      a BOUNDED interval between two roots: beyond the larger one, the value
      is valid again. A test only against the smaller root (an earlier
      approach) produces false positives there.
    * **A2 remainder.** (2 c_max)^5 / 5! <= rem_tol.

    The checks sit at c_max, not at the measured c -- the question is
    whether the BOUND is reliable, not whether the series holds at the
    current operating point.
    """
    out: list[str] = []
    c = cmax_skew(R, g1, g2, warn=False)

    if not (g1 <= 0.0 <= g2 or (g2 > 0.0 and g1 ** 2 < (56 / 27) * g2)):
        out.append(f"f not strictly monotonic (g1^2={g1**2:.4g} vs "
                   f"(56/27) g2={(56/27)*g2:.4g}) - further positive roots "
                   f"possible; smallest chosen")

    lnr = log_neff_ratio(c, g1, g2)
    if lnr > 0.0:
        out.append(f"c_max={c:.4f} violates N_eff <= n (log(N_eff/n)={lnr:.4g} > 0, "
                   f"Cauchy-Schwarz) - the value is meaningless there")

    rem = (2.0 * c) ** 5 / 120.0
    if rem > rem_tol:
        out.append(f"A2 remainder (2c)^5/5!={rem:.3g} > {rem_tol} - series no "
                   f"longer reliable at c_max")
    return out


def neff_ratio(w: np.ndarray) -> float:
    """Exact Kish N_eff/n — assumption-free, no series expansion."""
    s2 = float(np.sum(w ** 2))
    return float(w.sum() ** 2 / s2 / w.size) if s2 > 0 else float("nan")


def gewichte(dE: np.ndarray, beta: float) -> np.ndarray:
    """w = exp(-beta dE), stabilized.

    Subtracting the minimum makes the largest exponent exactly zero:
    overflow is thus ruled out, only underflow of the already negligible
    weights remains possible. N_eff is invariant to a constant offset in
    dE, so this intervention does not change the result.
    """
    return np.exp(-beta * (dE - dE.min()))


def _gpd_khat(x: np.ndarray) -> float:
    """Zhang & Stephens (2009), posterior mean of the GPD shape parameter."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 5 or x[-1] <= 0:
        return float("nan")
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs /= prior_bs * x[int(n / 4 + 0.5) - 1]
    bs += 1.0 / x[-1]
    ks = np.log1p(-bs[:, None] * x[None, :]).mean(axis=1)
    logl = n * (np.log(-bs / ks) - ks - 1.0)
    w = 1.0 / np.exp(logl - logl[:, None]).sum(axis=1)
    w /= w.sum()
    b_post = float((bs * w).sum())
    k_post = float(np.log1p(-b_post * x).mean())
    return float((n * k_post + prior_k * 0.5) / (n + prior_k))


def psis_khat(w: np.ndarray) -> float:
    """Pareto tail index of the weights. E[w^2] < infinity  <=>  khat < 0.5.

    Caution: a very noisy estimator, SE only decays as n^(-1/4). With a few
    hundred points, a single value cannot decide the 0.5 threshold --
    that's why khat is always computed on the FULL set here, never on a
    prefix.
    """
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    s = w.size
    if s < 25:
        return float("nan")
    n_tail = int(min(0.2 * s, 3.0 * np.sqrt(s)))
    if n_tail < 5:
        return float("nan")
    ws = np.sort(w)
    ueber = ws[-n_tail:] - ws[-n_tail - 1]
    ueber = ueber[ueber > 0]
    return 0.0 if ueber.size < 5 else _gpd_khat(ueber)


# --------------------------------------------------------------------------
# Sequential monitor
# --------------------------------------------------------------------------
def checkpoint_grid(n: int, first_frac: float = FIRST_FRAC,
                    ratio: float = 1.4) -> np.ndarray:
    """Geometric checkpoint grid from first_frac*n to n.

    Rule of thumb: first look at 10% of the points planned anyway. This is
    scale-free (n=500 -> starting at 50, n=5000 -> starting at 500) and
    yields about eight looks in both cases. Looking earlier gains little:
    below that, not only is SE(c) large, but the bound itself is
    unreliable, and the savings would already be at 90% anyway.

    NOTE, two separate quantities. The grid start first_frac*n and the
    filter K_FLOOR are NOT the same thing. For n < 500 the grid start is
    below K_FLOOR and the filter kicks in; for n >= 500 it is above and the
    filter has no effect. Merging both into a single max(K_FLOOR, n//10)
    gives, for n=400, the grid [50, 70, 98, ...] instead of [56, 78, 109,
    ...] -- i.e. a FIRST LOOK earlier than validated.
    """
    k0 = max(int(round(first_frac * n)), 10)
    ks = [k0]
    while ks[-1] * ratio < n:
        ks.append(int(round(ks[-1] * ratio)))
    ks.append(int(n))
    return np.array(sorted(set(ks)))


def checkpoints_fuer(n: int, k_floor: int = K_FLOOR,
                     first_frac: float = FIRST_FRAC) -> list[int]:
    """The grid, filtered to k >= k_floor -- the same way monitor_split does it."""
    ck = checkpoint_grid(n, first_frac)
    return [int(k) for k in ck if k_floor <= k <= n]


def se_cmax_boot(dE_prefix: np.ndarray, R: float, beta: float,
                 B: int, rng: np.random.Generator) -> float:
    """SE(c_max) from B bootstrap resamples of the SAME k points.

    Cannot be done analytically: the delta method via dc_max/dgamma1 =
    c^3/f' has f'(c_max) in the denominator, which goes to zero for noisy
    gamma1 — measured to be 60% too high on average, with outliers up to a
    factor of 35.
    """
    k = dE_prefix.size
    idx = rng.integers(0, k, (B, k))
    proben = dE_prefix[idx]
    werte = np.empty(B)
    for i in range(B):
        _, g1, g2 = momente(proben[i], beta)
        werte[i] = cmax_skew(R, g1, g2, warn=False)   # silent in the hot path
    return float(werte.std())


def monitor_schritt(prefix: np.ndarray, R: float, beta: float, band: float,
                    B: int, rng: np.random.Generator) -> dict:
    """A single checkpoint step: all quantities at k = prefix.size.

    Rule:  c(k) - band*SE(c)  >  c_max(k) + band*SE(c_max)   =>  FAIL

    Factored out of monitor() so the same step can also be run individually
    (live mode, one call = one checkpoint) without re-running the whole
    history.
    """
    k = prefix.size
    c, g1, g2 = momente(prefix, beta)
    cm = cmax_skew(R, g1, g2, warn=False)
    s_c = se_c(c, g2, k)
    s_cm = se_cmax_boot(prefix, R, beta, B, rng)
    feuert = (c - band * s_c) > (cm + band * s_cm)
    return {"k": k, "c": c, "gamma1": g1, "gamma2": g2,
            "c_max": cm, "se_c": s_c, "se_c_max": s_cm,
            "abstand": c - cm, "band": band * (s_c + s_cm),
            "feuert": bool(feuert)}


def monitor(dE: np.ndarray, R: float, beta: float, k_floor: int,
            band: float, B: int, rng: np.random.Generator,
            first_frac: float = FIRST_FRAC) -> dict:
    """Sequential FAIL-only monitor over the checkpoint grid.

    One-sided: an early PASS saves nothing, since the weights are needed in
    full at the end anyway. Only an early FAIL saves computation time. All
    quantities at k use exclusively dE[:k] — the monitor does not see the
    future.

    Single-sequence version of uq_mace.screening.monitor_split. Two
    deliberate differences from the library, neither affecting the
    decision: there, many sequences run row-wise in parallel, and the
    bound comes from cmax_skew_vec (Newton) instead of cmax_skew
    (numpy.roots) — on a grid of 3721 parameter pairs, both agree to
    4.7e-15.

    Retrospective simulation over an already complete dE: for the live
    single-step version on a growing dE, see monitor_schritt().
    """
    n = dE.size
    schritte = []
    for k in checkpoints_fuer(n, k_floor, first_frac):
        schritt = monitor_schritt(dE[:k], R, beta, band, B, rng)
        schritte.append(schritt)
        if schritt["feuert"]:
            return {"gefeuert": True, "k_stop": k,
                    "gespart": 1.0 - k / n, "schritte": schritte}
    return {"gefeuert": False, "k_stop": None, "gespart": 0.0, "schritte": schritte}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _bericht_live(erg: dict) -> str:
    """Compact report for a single live checkpoint (see _live_schritt).

    Deliberately shows only what was actually computed at this k -- no
    khat, no exact N_eff/n, no remainder diagnostics. The full
    certification only comes with the call at n >= n_plan (bericht()).
    """
    g = erg["gesamt"]
    z = []
    z.append("=" * 62)
    z.append("  KISH SCREENING — live checkpoint")
    z.append("=" * 62)
    z.append(f"  Points              {g['n']} of {g['n_plan']} planned")
    z.append(f"  Temperature         {g['T']:.1f} K   (beta = {g['beta']:.3f} 1/eV)")
    z.append(f"  Target R            {g['R']}")
    schritte = erg["monitor"]["schritte"] if erg.get("monitor") else []
    if schritte:
        s = schritte[0]
        z.append("")
        z.append(f"  c = beta*std(dE)    {s['c']:.4f}")
        z.append(f"  c_max (skewed)      {s['c_max']:.4f}")
        z.append(f"  Gap c - c_max       {s['abstand']:+.4f}   (band {s['band']:.4f})")
    z.append("")
    z.append("=" * 62)
    z.append(f"  VERDICT: {erg['urteil']}")
    if erg.get("begruendung"):
        z.append(f"  {erg['begruendung']}")
    z.append("=" * 62)
    return "\n".join(z)


def bericht(erg: dict, zeige_schritte: bool) -> str:
    if erg["gesamt"].get("live"):
        return _bericht_live(erg)
    z = []
    g = erg["gesamt"]
    z.append("=" * 62)
    z.append("  KISH SCREENING — does the reweighting hold?")
    z.append("=" * 62)
    z.append(f"  Points n            {g['n']}")
    z.append(f"  Temperature         {g['T']:.1f} K   (beta = {g['beta']:.3f} 1/eV)")
    z.append(f"  Target R            {g['R']}")
    z.append("")
    z.append(f"  std(dE)             {g['sigma']*1000:.3f} meV")
    z.append(f"  c = beta*std(dE)    {g['c']:.4f}")
    z.append(f"  Skewness  gamma1    {g['gamma1']:+.4f}")
    z.append(f"  Kurtosis  gamma2    {g['gamma2']:+.4f}")
    z.append("")
    z.append(f"  c_max (Gaussian)    {g['c_max_gauss']:.4f}")
    z.append(f"  c_max (skewed)      {g['c_max']:.4f}   <- used")
    z.append(f"  rho = c/c_max       {g['rho']:.3f}")
    z.append("")
    z.append(f"  N_eff/n (exact)     {g['neff_ratio']:.4f}"
             f"   {'>=' if g['neff_ratio'] >= g['R'] else '<'} R = {g['R']}")
    z.append(f"  N_eff/n (series)    {g['neff_ratio_reihe']:.4f}"
             f"   Gaussian alone: {g['neff_ratio_gauss']:.4f}")
    if np.isnan(g['khat']):
        khat_status = "not determinable (< 25 positive weights)"
    elif g['khat'] < KHAT_GATE:
        khat_status = "gate passed"
    else:
        khat_status = "GATE VIOLATED"
    z.append(f"  khat (tail index)   {g['khat']:+.3f}   {khat_status}")
    z.append(f"  Remainder (2c)^5/5! {g['r5']:.4f}"
             f"   {'ok' if g['r5'] <= R5_TOL else 'series unusable'}")
    for h in erg.get("hinweise", []):
        z.append(f"  [Note] {h}")

    m = erg.get("monitor")
    if m is not None:
        z.append("")
        z.append("-" * 62)
        z.append("  SEQUENTIAL MONITOR")
        z.append("-" * 62)
        z.append(f"  Rule: c(k) - {g['band']:g}*SE(c) > c_max(k) + "
                 f"{g['band']:g}*SE(c_max)")
        z.append(f"  Grid: from {g['first_frac']:.0%} of n, factor 1.4, "
                 f"filtered to k >= {m['k_floor']}")
        z.append(f"  Checkpoints: {m['checkpoints']}")
        if zeige_schritte:
            z.append("")
            z.append(f"  {'k':>7}{'c':>9}{'c_max':>9}{'Gap':>10}"
                     f"{'Band':>9}{'Verdict':>10}")
            z.append("  " + "-" * 52)
            for s in m["schritte"]:
                z.append(f"  {s['k']:>7}{s['c']:>9.4f}{s['c_max']:>9.4f}"
                         f"{s['abstand']:>+10.4f}{s['band']:>9.4f}"
                         f"{'FAIL' if s['feuert'] else 'continue':>10}")
        z.append("")
        if m["gefeuert"]:
            z.append(f"  -> Stop at k = {m['k_stop']} of {g['n']}")
            z.append(f"     {m['gespart']*100:.0f}% of the DFT points would have been saved.")
        else:
            z.append("  -> no stop. The monitor would have let the run finish.")

    z.append("")
    z.append("=" * 62)
    z.append(f"  VERDICT: {erg['urteil']}")
    if erg.get("begruendung"):
        z.append(f"  {erg['begruendung']}")
    z.append("=" * 62)
    return "\n".join(z)


# --------------------------------------------------------------------------
# Main program
# --------------------------------------------------------------------------
def parser_bauen() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kish_screening.py",
        description="Checks whether thermodynamic reweighting statistically "
                    "holds up, and simulates the sequential stopping monitor.",
        epilog="Exit codes: 0 PASS | 1 FAIL | 2 usage | 3 data | 4 unreliable",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dft", metavar="DFT_FILE",
                   help="reference energies (DFT). If the file contains "
                        "both 'e_dft' and 'e_mace'/'energies' (project "
                        "cache), ML_FILE can be omitted.")
    p.add_argument("ml", metavar="ML_FILE", nargs="?",
                   help="model energies (MACE or similar)")
    p.add_argument("-R", "--target", type=float, default=0.8, metavar="F",
                   help="required N_eff/n, in (0,1) (default 0.8)")
    p.add_argument("-T", "--temperature", type=float, default=292.0, metavar="K",
                   help="temperature in Kelvin (default 292)")
    p.add_argument("-u", "--units", default="eV", choices=sorted(UNITS),
                   help="unit of the input energies (default eV)")
    p.add_argument("--key-dft", metavar="NAME", help="npz key of the DFT file")
    p.add_argument("--key-ml", metavar="NAME", help="npz key of the ML file")
    p.add_argument("-k", "--k-floor", type=int, default=K_FLOOR, metavar="N",
                   help=f"checkpoints below k are discarded "
                        f"(default {K_FLOOR})")
    p.add_argument("-N", "--n-plan", type=int, default=None, metavar="N",
                   help="planned total budget of the campaign. Without this "
                        "option, the checkpoint grid runs relative to the "
                        "currently available number of points (batch mode). "
                        "Required with --live: fixes the grid regardless "
                        "of how many points are currently available.")
    p.add_argument("--live", action="store_true",
                   help="live checkpoint mode for embedding into a running "
                        "campaign: checks only the ONE checkpoint currently "
                        "due at k = current point count (no grid walk), "
                        "instead of re-simulating the history. Requires "
                        "--n-plan. Below --k-floor: CONTINUE without a "
                        "check. At current point count >= --n-plan: full "
                        "certification as in batch mode.")
    p.add_argument("--first-frac", type=float, default=FIRST_FRAC, metavar="F",
                   help=f"grid start as a fraction of n (default {FIRST_FRAC}). "
                        f"Do not confuse with --k-floor: the grid starts at "
                        f"first_frac*n, k_floor cuts it off afterwards.")
    p.add_argument("-b", "--band", type=float, default=1.0, metavar="F",
                   help="bandwidth in standard errors per side (default 1.0)")
    p.add_argument("-B", "--bootstrap", type=int, default=200, metavar="N",
                   help="resamples per checkpoint for SE(c_max) (default 200)")
    p.add_argument("--seed", type=int, default=0, metavar="N",
                   help="random seed, for reproducible runs (default 0)")
    p.add_argument("--no-monitor", action="store_true",
                   help="only the metrics, no checkpoint simulation")
    p.add_argument("--steps", action="store_true",
                   help="print a table of all checkpoints")
    p.add_argument("--json", action="store_true",
                   help="result as JSON on stdout (for further processing)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="only the verdict (PASS/FAIL/UNCLEAR)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _live_schritt(dE: np.ndarray, args, beta: float, k_floor: int) -> tuple[dict, int]:
    """One live checkpoint: exactly ONE call to monitor_schritt() at k = dE.size.

    No grid walk, no khat gate, no remainder/C_VALID diagnostics -- per
    analyses/13_sequential_screening/README.md these are only meant as a
    follow-up check on the PASS branch and for the final certification
    anyway (see rechnen(), branch n >= n_plan). cmax_skew's c_hi fallback
    keeps the FAIL rule conservative even outside the series' validity
    range.
    """
    n = dE.size
    gesamt = {"n": n, "n_plan": args.n_plan, "T": args.temperature, "beta": beta,
              "R": args.target, "units": args.units, "band": args.band,
              "k_floor": k_floor, "live": True}
    erg = {"gesamt": gesamt, "version": __version__, "hinweise": [], "warnungen": []}

    if n < k_floor:
        erg["urteil"] = "CONTINUE"
        erg["begruendung"] = (f"{n} of {args.n_plan} points -- still below "
                              f"--k-floor={k_floor}, no check.")
        erg["monitor"] = None
        return erg, EXIT_PASS

    rng = np.random.default_rng(args.seed)
    schritt = monitor_schritt(dE, args.target, beta, args.band, args.bootstrap, rng)
    gesamt.update({"c": schritt["c"], "gamma1": schritt["gamma1"],
                   "gamma2": schritt["gamma2"], "c_max": schritt["c_max"]})
    erg["monitor"] = {"gefeuert": schritt["feuert"],
                      "k_stop": schritt["k"] if schritt["feuert"] else None,
                      "schritte": [schritt]}

    if schritt["feuert"]:
        erg["urteil"] = "FAIL"
        erg["begruendung"] = (f"Monitor fires at k={n}: c={schritt['c']:.4f} "
                              f"- {args.band:g}*SE > c_max={schritt['c_max']:.4f} "
                              f"+ {args.band:g}*SE")
        return erg, EXIT_FAIL

    erg["urteil"] = "CONTINUE"
    erg["begruendung"] = (f"{n} of {args.n_plan} points, no FAIL "
                          f"(gap {schritt['abstand']:+.4f}) -- continue the campaign.")
    return erg, EXIT_PASS


def rechnen(args) -> tuple[dict, int]:
    if not 0.0 < args.target < 1.0:
        raise Abbruch(EXIT_USAGE, f"R must be in (0,1), got {args.target}")
    if args.temperature <= 0:
        raise Abbruch(EXIT_USAGE, f"temperature must be positive, got {args.temperature}")
    if args.band < 0:
        raise Abbruch(EXIT_USAGE, f"bandwidth must not be negative, got {args.band}")
    if args.bootstrap < 20:
        raise Abbruch(EXIT_USAGE, f"--bootstrap should be >= 20, got {args.bootstrap}")

    faktor = UNITS[args.units]
    if args.ml is None:
        e_dft, e_ml = lade_paar(args.dft)
        e_dft, e_ml = e_dft * faktor, e_ml * faktor
    else:
        e_dft = lade_energien(args.dft, args.key_dft) * faktor
        e_ml = lade_energien(args.ml, args.key_ml) * faktor

    k_floor = args.k_floor
    if k_floor < 5:
        raise Abbruch(EXIT_USAGE, f"--k-floor must be >= 5, got {k_floor}")
    if not 0.0 < args.first_frac <= 1.0:
        raise Abbruch(EXIT_USAGE,
                      f"--first-frac must be in (0,1], got {args.first_frac}")
    if args.live and args.n_plan is None:
        raise Abbruch(EXIT_USAGE, "--live requires --n-plan")
    if args.n_plan is not None and args.n_plan < k_floor:
        raise Abbruch(EXIT_USAGE,
                      f"--n-plan must be >= --k-floor, got {args.n_plan} < {k_floor}")

    # only pass n_plan through to validation in live mode: the batch path
    # still builds the monitor over the actual dE.size (see below); an
    # n_plan there would be a window with no effect, silently defeating
    # the "too few points" check.
    dE = pruefe_daten(e_dft, e_ml, k_floor, n_plan=args.n_plan if args.live else None)
    beta = 1.0 / (KB_EV * args.temperature)
    n = dE.size

    if args.live and n < args.n_plan:
        return _live_schritt(dE, args, beta, k_floor)

    c, g1, g2 = momente(dE, beta)
    cm = cmax_skew(args.target, g1, g2, warn=False)
    w = gewichte(dE, beta)
    kh = psis_khat(w)
    r5 = (2.0 * c) ** 5 / 120.0
    lnr_reihe = log_neff_ratio(c, g1, g2)

    gesamt = {"n": n, "T": args.temperature, "beta": beta, "R": args.target,
              "units": args.units, "band": args.band,
              "sigma": float(dE.std(ddof=1)), "c": c, "gamma1": g1, "gamma2": g2,
              "c_max": cm, "c_max_gauss": cmax_gauss(args.target),
              "rho": c / cm, "neff_ratio": neff_ratio(w), "khat": kh, "r5": r5,
              "neff_ratio_reihe": float(np.exp(lnr_reihe)),
              "neff_ratio_gauss": float(np.exp(-c ** 2)),
              "k_floor": k_floor, "first_frac": args.first_frac, "n_plan": args.n_plan,
              "diagnose": diagnose(args.target, g1, g2)}
    erg = {"gesamt": gesamt, "version": __version__}

    # --- reliability first: without it neither PASS nor FAIL has meaning ---
    warnungen = []
    if not np.isnan(kh) and kh >= KHAT_GATE:
        warnungen.append(
            f"khat = {kh:.3f} >= {KHAT_GATE}: E[w^2] does not exist, N_eff "
            f"has no population limit. The computed value is a sample "
            f"statistic without a target.")
    if c >= C_VALID:
        warnungen.append(
            f"c = {c:.4f} >= C_VALID = {C_VALID:.4f}: the cumulant series is "
            f"worthless here, c_max cannot be determined.")
    elif r5 > R5_TOL:
        warnungen.append(
            f"Remainder (2c)^5/5! = {r5:.4f} > {R5_TOL}: the series no "
            f"longer holds reliably at this c.")
    # preconditions of the quartic -- at c_max, not at the measured c.
    # Non-monotonicity is a NOTE, not an error: cmax_skew takes the
    # smallest positive root, and that is the correct one. It should still
    # be visible though. The other two messages invalidate the bound.
    hinweise = [m for m in gesamt["diagnose"] if "monoton" in m]
    warnungen.extend(f"Bound: {m}" for m in gesamt["diagnose"]
                     if "monoton" not in m)
    erg["hinweise"] = hinweise
    erg["warnungen"] = warnungen

    if not args.no_monitor:
        rng = np.random.default_rng(args.seed)
        m = monitor(dE, args.target, beta, k_floor, args.band, args.bootstrap,
                    rng, args.first_frac)
        m["checkpoints"] = [s["k"] for s in m["schritte"]]
        m["k_floor"] = k_floor
        erg["monitor"] = m

    # --- verdict ---------------------------------------------------------
    exakt_fail = gesamt["neff_ratio"] < args.target
    monitor_fail = erg.get("monitor", {}).get("gefeuert", False)

    if warnungen and not exakt_fail:
        erg["urteil"] = "UNCLEAR"
        erg["begruendung"] = warnungen[0]
        return erg, EXIT_UNRELIABLE
    if exakt_fail or monitor_fail:
        erg["urteil"] = "FAIL"
        teile = []
        if exakt_fail:
            teile.append(f"N_eff/n = {gesamt['neff_ratio']:.4f} < R = {args.target}")
        if monitor_fail:
            teile.append(f"Monitor fires at k = {erg['monitor']['k_stop']}")
        erg["begruendung"] = "; ".join(teile)
        return erg, EXIT_FAIL
    erg["urteil"] = "PASS"
    erg["begruendung"] = (f"N_eff/n = {gesamt['neff_ratio']:.4f} >= R = {args.target}, "
                          f"rho = {gesamt['rho']:.3f}")
    return erg, EXIT_PASS


def main(argv=None) -> int:
    args = parser_bauen().parse_args(argv)
    try:
        erg, code = rechnen(args)
    except Abbruch as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        return 130

    if args.json:
        print(json.dumps(erg, indent=2, ensure_ascii=False, default=float))
    elif args.quiet:
        print(erg["urteil"])
    else:
        print(bericht(erg, args.steps))
    for w in erg.get("warnungen", []):
        print(f"WARNING: {w}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
