"""Sequenzielles Screening: Entscheidungsschranke, laufende Momente, FAIL-Monitor.

Der Workflow ueberwacht die Gewichte w_i eines Kandidatenmodells, waehrend sie
anfallen, und bricht ab, sobald statistisch klar ist, dass das Reweighting nicht
tragen wird (N_eff/n < R). Die Entscheidung laeuft ueber

    c = beta * std(dE)   gegen   c_max^schief(gamma1, gamma2, R)

Zwei Dinge, die den Entwurf bestimmen:

1. **Einseitig.** Ein frueher PASS spart nichts -- die Gewichte werden am Ende
   ohnehin vollstaendig gebraucht. Nur ein frueher FAIL spart Rechenzeit. Damit
   bleibt genau eine Fehlerart: ein brauchbares Modell abbrechen.

2. **Kausal.** Jede Groesse bei Punkt k benutzt ausschliesslich dE[:k] -- auch
   gamma1, gamma2 und die daraus gebildete Schranke.

Alle Funktionen ziehen beta und R aus einem Modulkontext, der einmal gesetzt wird:

    >>> from uq_mace import screening as scr
    >>> scr.configure(beta=39.742, R=0.8)
    >>> k, c, g1, g2 = scr.run_stats(dE)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .reweighting import running_moments

__all__ = [
    "configure", "context",
    "K_FLOOR", "Q_ALPHA", "CHECKPOINTS", "checkpoint_grid", "C_VALID",
    "cmax_gauss", "cmax_skew", "cmax_skew_vec", "log_neff_ratio",
    "run_stats", "run_stats_2d", "se_c",
    "decide_naive", "fail_only_2d", "stat_D", "monitor_boot",
]

# ---------------------------------------------------------------------------
# Voreinstellungen
# ---------------------------------------------------------------------------
K_FLOOR = 50         # erster Blick; Faustregel: 10 % der geplanten Punkte (siehe
                     # checkpoint_grid). Darunter ist nicht nur SE(c) gross, sondern
                     # auch die SCHRANKE unzuverlaessig geschaetzt.
R5_TOL  = 0.05       # zulaessiges Restglied (2c)^5/5! der Kumulantenreihe
C_VALID = 0.5 * (120.0 * R5_TOL) ** 0.2     # ~0.716: darueber ist die Reihe wertlos
Q_ALPHA = 1.64       # Quantil der Standardnormalverteilung (einseitig 5 %).
                     # NICHT zu verwechseln mit dem standardisierten z aus der
                     # Herleitung -- siehe Notebook §2.2.
def checkpoint_grid(n: int, first_frac: float = 0.10, ratio: float = 1.4):
    """Geometrisches Checkpoint-Raster ab first_frac*n bis n.

    Faustregel: der erste Blick bei 10 % der ohnehin geplanten Punkte. Das ist
    skalenfrei (n=500 -> ab 50, n=5000 -> ab 500) und liefert in beiden Faellen
    rund acht Blicke. Frueher zu schauen bringt wenig: unterhalb davon ist nicht
    nur SE(c) gross, sondern auch die Schranke selbst unzuverlaessig (siehe
    cmax_skew_vec), und die Ersparnis waere ohnehin schon bei 90 %.
    """
    k0 = max(int(round(first_frac * n)), 10)
    ks = [k0]
    while ks[-1] * ratio < n:
        ks.append(int(round(ks[-1] * ratio)))
    ks.append(int(n))
    return np.array(sorted(set(ks)))


CHECKPOINTS = checkpoint_grid(500)

_CTX: dict[str, float | None] = {"beta": None, "R": None}


def configure(beta: float, R: float = 0.8) -> dict:
    """beta [1/eV] und das Effizienzziel R einmalig setzen."""
    _CTX["beta"] = float(beta)
    _CTX["R"] = float(R)
    return context()


def context() -> dict:
    """Aktueller Kontext plus die abgeleiteten Groessen."""
    if _CTX["beta"] is None:
        raise RuntimeError("screening.configure(beta, R) zuerst aufrufen.")
    R = _CTX["R"]
    return {"beta": _CTX["beta"], "R": R,
            "LNR": float(np.log(R)), "cmax_gauss": cmax_gauss(R)}


def _bR() -> tuple[float, float, float]:
    c = context()
    return c["beta"], c["R"], c["LNR"]


# ---------------------------------------------------------------------------
# Schranken
# ---------------------------------------------------------------------------
def cmax_gauss(R: float) -> float:
    """Gauss-Schranke: N_eff/n = exp(-c^2) >= R  <=>  c <= sqrt(-ln R)."""
    return float(np.sqrt(-np.log(R)))


def cmax_skew(R: float, g1: float, g2: float, c_hi: float | None = None,
              warn: bool = True) -> float:
    """Kleinste positive Nullstelle von c^2 - g1 c^3 + 7/12 g2 c^4 = -ln R.

    Skalares Gegenstueck zu cmax_skew_vec, mit DENSELBEN zwei Absicherungen:

    * Gesucht wird nur bis C_VALID -- darueber ist die abgeschnittene Reihe
      wertlos (Restglied (2c)^5/5! > R5_TOL), und eine dort gefundene Wurzel
      waere ein Artefakt ihrer Divergenz.
    * Kein Vorzeichenwechsel im gueltigen Bereich -> Rueckfall auf die OBERE
      Grenze, nicht auf die Gauss-Schranke. Letztere liegt TIEFER als die wahre
      Schranke und liesse den FAIL-only-Monitor zu frueh feuern.

    Kein naives Bracketing ueber das ganze Intervall: fuer gamma2 < 0 kippt f(c)
    bei grossem c wieder unter null, ein Bracket [0, c_hi] haette dann u.U. keinen
    Vorzeichenwechsel, obwohl die gesuchte kleinste Wurzel existiert. Deshalb wird
    der erste Aufwaertsdurchgang auf einem Gitter gesucht und nur dort gebracket.
    """
    c_hi = C_VALID if c_hi is None else c_hi
    target = -np.log(R)

    def f(c):
        return c**2 - g1 * c**3 + (7 / 12) * g2 * c**4 - target

    grid = np.linspace(1e-4, c_hi, 2000)
    fv = f(grid)
    up = np.where((fv[:-1] < 0) & (fv[1:] >= 0))[0]
    if up.size == 0:
        if warn:
            print(f"    [Hinweis] cmax_skew: keine Wurzel unterhalb C_VALID fuer "
                  f"g1={g1:+.3f}, g2={g2:+.3f} -> konservativer Rueckfall {c_hi:.3f}")
        return float(c_hi)
    return float(brentq(f, grid[up[0]], grid[up[0] + 1]))


def cmax_skew_vec(g1, g2, iters: int = 8):
    """cmax_skew fuer ganze Arrays -- Newton ab der Gauss-Loesung, auf den
    Gueltigkeitsbereich der Reihe beschraenkt.

    Zwei Fallen, die beide real auftreten, wenn gamma1, gamma2 aus wenigen
    Punkten geschaetzt sind:

    (1) Newton konvergiert nicht. Frueher wurde dann auf die Gauss-Schranke
        zurueckgefallen -- die liegt aber TIEFER als die wahre Schranke, und eine
        zu tiefe Schranke laesst den FAIL-only-Monitor zu frueh feuern. Falsche
        Richtung.
    (2) Bei stark ueberschaetztem gamma1 (Stichprobe streut bei k=20 zwischen
        -0.7 und +1.7, wahr +0.5) hat die Quartik im sinnvollen Bereich GAR KEINE
        Wurzel: die Schiefekorrektur haelt log(N_eff/n) bis c ~ 2.4 ueber ln R.
        Newton findet diese Wurzel korrekt -- nur ist die Reihe dort sinnlos
        (Restglied ~ 2000).

    Beides sind Symptome davon, die abgeschnittene Quartik ausserhalb ihres
    Gueltigkeitsbereichs auszuwerten. Aus dem Restglied (2c)^5/5! <= R5_TOL folgt
    c <= C_VALID ~ 0.716. Ausserhalb wird nicht extrapoliert; existiert dort keine
    Wurzel, ist die konservative Antwort die OBERE Grenze (hohe Schranke -> nicht
    feuern), nicht die Gauss-Schranke.
    """
    _, R, LNR = _bR()
    c0 = cmax_gauss(R)
    if c0 > C_VALID:
        raise ValueError(f"R={R} zu klein: c_max^Gauss={c0:.3f} liegt bereits ausserhalb "
                         f"des Gueltigkeitsbereichs der Reihe (C_VALID={C_VALID:.3f}).")
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    c = np.full(np.broadcast(g1, g2).shape, c0, dtype=float)
    for _ in range(iters):
        f = c**2 - g1 * c**3 + (7 / 12) * g2 * c**4 + LNR
        fp = 2 * c - 3 * g1 * c**2 + (7 / 3) * g2 * c**3
        c = np.clip(c - np.where(np.abs(fp) > 1e-12, f / fp, 0.0), 1e-3, C_VALID)
    resid = c**2 - g1 * c**3 + (7 / 12) * g2 * c**4 + LNR
    ok = (np.abs(resid) < 1e-6) & (c < C_VALID - 1e-9)
    return np.where(ok, c, C_VALID)      # kein verlaesslicher Wert -> nicht feuern


def log_neff_ratio(c, g1, g2):
    """log(N_eff/n) nach der Kumulantenentwicklung: -c^2 + g1 c^3 - 7/12 g2 c^4."""
    return -c**2 + g1 * c**3 - (7 / 12) * g2 * c**4


# ---------------------------------------------------------------------------
# Laufende Momente
# ---------------------------------------------------------------------------
def run_stats(dE):
    """Laufende (k, c, gamma1, gamma2) nach jedem neuen Punkt."""
    beta, _, _ = _bR()
    m = running_moments(np.asarray(dE, dtype=float))
    return (m["k"], beta * np.nan_to_num(m["std"]),
            np.nan_to_num(m["skew"]), np.nan_to_num(m["kurtosis"]))


def run_stats_2d(D):
    """Zeilenweises Pendant zu run_stats, ueber Achse 1 vektorisiert.

    Gleiche Formeln; unterhalb von k=4 liefert running_moments bewusst NaN,
    run_stats_2d dagegen den Rohwert -- unterhalb K_FLOOR ohne Bedeutung.
    """
    beta, _, _ = _bR()
    D = D - D.mean(axis=1, keepdims=True)
    k = np.arange(1, D.shape[1] + 1)
    s1 = np.cumsum(D, 1); s2 = np.cumsum(D**2, 1)
    s3 = np.cumsum(D**3, 1); s4 = np.cumsum(D**4, 1)
    mu = s1 / k
    m2 = np.maximum(s2 / k - mu**2, 0.0)
    m3 = s3 / k - 3 * mu * (s2 / k) + 2 * mu**3
    m4 = s4 / k - 4 * mu * (s3 / k) + 6 * mu**2 * (s2 / k) - 3 * mu**4
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(np.where(k > 1, m2 * k / np.maximum(k - 1, 1), np.nan))
        g1 = np.where(m2 > 0, m3 / m2**1.5, np.nan)
        g2 = np.where(m2 > 0, m4 / m2**2 - 3.0, np.nan)
    return k, beta * np.nan_to_num(std), np.nan_to_num(g1), np.nan_to_num(g2)


def se_c(c, g2, k):
    """SE(c)/c = sqrt((gamma2 + 2) / 4k). gamma2 >= -2 ist eine
    Verteilungsschranke; der Ausdruck wird hier trotzdem abgesichert, weil die
    Stichproben-Kurtosis bei sehr kleinem k numerisch darunter rutschen kann."""
    return c * np.sqrt(np.maximum(g2 + 2.0, 0.0) / (4.0 * np.maximum(k, 2)))


# ---------------------------------------------------------------------------
# Entscheidungsregeln
# ---------------------------------------------------------------------------
def decide_naive(c, k, g1, g2, q: float = Q_ALPHA, schranke: str = "skew"):
    """Zweiseitig, bei jedem k: das q-Band um c gegen die Schranke.

    Historische Vergleichsfassung -- der Workflow benutzt monitor_boot.
    """
    _, R, _ = _bR()
    se = se_c(c, g2, k)
    cm = cmax_skew_vec(g1, g2) if schranke == "skew" else cmax_gauss(R)
    out = np.full(np.shape(c), "WEITER", dtype=object)
    out[c + q * se < cm] = "PASS"
    out[c - q * se > cm] = "FAIL"
    return out


def fail_only_2d(D, q: float = Q_ALPHA, schranke: str = "skew",
                 k_floor: int = K_FLOOR):
    """Einseitig, bei jedem k, analytisches Band. Vergleichsfassung zu monitor_boot."""
    _, R, _ = _bR()
    k, c, g1, g2 = run_stats_2d(D)
    c_lo = c - q * se_c(c, g2, k)
    cm = cmax_gauss(R) if schranke == "gauss" else cmax_skew_vec(g1, g2)
    fail = (c_lo > cm) & (k >= k_floor)
    a = fail.any(1)
    return a, np.where(a, k[np.argmax(fail, 1)], -1)


def stat_D(X):
    """Entscheidungsgroesse D = c - c_max je Zeile von X (n, k), plus c, g1, g2."""
    beta, _, _ = _bR()
    k = X.shape[1]
    Y = X - X.mean(1, keepdims=True)
    m2 = (Y**2).mean(1); m3 = (Y**3).mean(1); m4 = (Y**4).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = beta * np.sqrt(np.maximum(m2 * k / max(k - 1, 1), 0.0))
        g1 = np.where(m2 > 0, m3 / m2**1.5, 0.0)
        g2 = np.where(m2 > 0, m4 / m2**2 - 3.0, 0.0)
    return c - cmax_skew_vec(g1, g2), c, g1, g2


def monitor_boot(D_seq, q: float = Q_ALPHA, B: int = 100, rng=None,
                 chunk: int = 250, checkpoints=None, k_floor: int = K_FLOOR):
    """Die verwendete Regel: FAIL sobald D(k) > q * SE_boot(D(k)).

    Geprueft wird nur an CHECKPOINTS (geometrisches Raster statt jedem k, das
    senkt die Zahl der Blicke). Die Bandbreite wird gebootstrappt statt
    analytisch fortgepflanzt, weil auch c_max geschaetzt ist und die
    Normaltheorie-Formeln fuer gamma1, gamma2 bei kleinem k nicht taugen.

    PASS wird nie behauptet. Rueckgabe: (gefeuert, k_bei_dem_gefeuert),
    k = -1 heisst 'nie gefeuert'.

    B: Bootstrap-Resamples je Checkpoint. Bei B=100 streut SE_boot(D) selbst um
    rund 9 %, das effektive q also um +-0.14. Wo die Sequenzzahl klein ist (die
    Modell-Laeufe), lohnt B=300; fuer die grossen Sweeps waere das zu teuer.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    # Raster passend zur tatsaechlichen Sequenzlaenge, nicht das globale Default
    ck = checkpoint_grid(D_seq.shape[1]) if checkpoints is None else np.asarray(checkpoints)
    ck = ck[(ck <= D_seq.shape[1]) & (ck >= k_floor)]
    n = D_seq.shape[0]
    fired = np.zeros(n, dtype=bool)
    kfire = np.full(n, -1)
    for k in ck:
        todo = np.where(~fired)[0]
        if todo.size == 0:
            break
        Dk = stat_D(D_seq[todo, :k])[0]
        se = np.empty(todo.size)
        for a in range(0, todo.size, chunk):          # gechunkt, sonst Speicher
            b = min(a + chunk, todo.size)
            sub = D_seq[todo[a:b], :k]
            idx = rng.integers(0, k, (b - a, B, k))
            boot = np.take_along_axis(sub[:, None, :], idx, axis=2)
            se[a:b] = stat_D(boot.reshape(-1, k))[0].reshape(b - a, B).std(1)
        sel = todo[Dk > q * se]
        fired[sel] = True
        kfire[sel] = k
    return fired, kfire
