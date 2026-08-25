#!/usr/bin/env python3
"""
kish_screening.py — lohnt sich das Reweighting?
====================================================

Entscheidet fuer einen Satz DFT- und ML-Energien, ob thermodynamisches
Reweighting statistisch tragen wird, und simuliert den sequenziellen Monitor:
nach wie vielen Punkten kann man den Lauf abbrechen?

    python3 kish_screening.py DFT_DATEI ML_DATEI [Optionen]

Exit-Codes (fuer die Benutzung in Shell-Skripten):

    0   PASS  — Kriterium erfuellt, Reweighting traegt (im Live-Modus auch:
        WEITER, kein FAIL bisher, Kampagne fortsetzen)
    1   FAIL  — Abbruchbedingung erfuellt, Rechnung einstellen
    2   Aufrufsfehler (Argumente, Datei fehlt, unbekanntes Format)
    3   Datenfehler (ungleiche Laenge, zu wenige Punkte, NaN)
    4   Ergebnis nicht belastbar (Momentbedingung verletzt oder Reihe
        ausserhalb ihres Gueltigkeitsbereichs) — weder PASS noch FAIL

Beispiel (Batch, fertiger Datensatz):

    Zwei Dateien input:

    python3 kish_screening.py e_dft.npy e_mace.npy -R 0.8 || {
        echo "Reweighting traegt nicht — MD abbrechen" >&2
        exit 1
    }

    Ein Dateien Input:
    python3 kish_screening.py cache/single_mace-L0-01_testfull_n5000.npz \ -R 0.8 -T 292 --steps
    echo "Exit-Code: $?"

Beispiel (Live, Einbettung in eine laufende MD/DFT-Kampagne mit geplant 5000
Punkten): bei jedem neu hinzugekommenen DFT-Batch neu aufrufen, mit den bis
dahin gesammelten Punkten. Ein einzelner Aufruf prueft NUR den gerade
faelligen Checkpoint (kein Raster-Walk) und gibt WEITER/FAIL zurueck; sobald
alle 5000 Punkte vorliegen, liefert derselbe Aufruf automatisch die volle
Zertifizierung (khat, exaktes N_eff/n, PASS/FAIL/UNKLAR):

    python3 kish_screening.py e_dft_bisher.npy e_mace_bisher.npy \
        -R 0.8 -T 292 -N 5000 --live -q || {
        echo "FAIL — MD/DFT-Kampagne abbrechen" >&2
        exit 1
    }

Nur numpy notwendig.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

__version__ = "1.1"

# --------------------------------------------------------------------------
# Konstanten
# --------------------------------------------------------------------------
KB_EV = 8.617333262e-5      # Boltzmann-Konstante in eV/K
R5_TOL = 0.05               # zulaessiges Restglied (2c)^5/5! der Kumulantenreihe
C_VALID = 0.5 * (120.0 * R5_TOL) ** 0.2      # ~0.7155, darueber ist die Reihe wertlos
KHAT_GATE = 0.5             # E[w^2] < unendlich  <=>  khat < 0.5

K_FLOOR = 50                # frueheste Auswertung; darunter ist nicht nur SE(c)
                            # gross, sondern die SCHRANKE selbst unzuverlaessig
                            # geschaetzt (bei k=5 hat die Quartik in 43 % der
                            # Ziehungen gar keine Wurzel im Gueltigkeitsbereich)
FIRST_FRAC = 0.10           # Rasteranfang als Anteil von n -- NICHT identisch
                            # mit K_FLOOR, siehe checkpoint_grid

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_DATA, EXIT_UNRELIABLE = 0, 1, 2, 3, 4

UNITS = {"eV": 1.0, "meV": 1e-3, "Ha": 27.211386245988,
         "Ry": 13.605693122994, "kcal/mol": 0.0433641153087705,
         "kJ/mol": 0.010364269656262}


class Abbruch(Exception):
    """Kontrollierter Abbruch mit Exit-Code."""

    def __init__(self, code: int, text: str):
        super().__init__(text)
        self.code = code


# --------------------------------------------------------------------------
# Einlesen
# --------------------------------------------------------------------------
def lade_energien(pfad: str, key: str | None = None) -> np.ndarray:
    """Energien aus .npz, .npy, .txt/.dat/.csv lesen. Rueckgabe: 1D float-Array.

    npz: nimmt `key`, falls angegeben. Sonst wird der erste passende Schluessel
    aus einer Vorzugsliste gesucht; hat der Treffer zwei Achsen (Komitee), wird
    ueber die erste (Member-Achse) gemittelt.
    """
    p = Path(pfad)
    if not p.exists():
        raise Abbruch(EXIT_USAGE, f"Datei nicht gefunden: {p}")
    if p.is_dir():
        raise Abbruch(EXIT_USAGE, f"ist ein Verzeichnis, keine Datei: {p}")

    endung = p.suffix.lower()
    try:
        if endung == ".npz":
            d = np.load(p, allow_pickle=False)
            if key is not None:
                if key not in d.files:
                    raise Abbruch(EXIT_USAGE,
                                  f"{p.name}: Schluessel '{key}' fehlt. "
                                  f"Vorhanden: {', '.join(d.files)}")
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
                                      f"{p.name}: kein eindeutiger Energie-Schluessel. "
                                      f"Vorhanden: {', '.join(d.files)}. "
                                      f"Mit --key-dft / --key-ml angeben.")
                a = d[treffer]
        elif endung == ".npy":
            a = np.load(p, allow_pickle=False)
        elif endung in (".txt", ".dat", ".csv", ".tsv", ""):
            trenn = "," if endung == ".csv" else None
            a = np.loadtxt(p, delimiter=trenn, comments=("#", "!", "%"))
        else:
            raise Abbruch(EXIT_USAGE,
                          f"{p.name}: unbekannte Endung '{endung}'. "
                          f"Unterstuetzt: .npz .npy .txt .dat .csv .tsv")
    except Abbruch:
        raise
    except Exception as e:                                   # noqa: BLE001
        raise Abbruch(EXIT_USAGE, f"{p.name}: nicht lesbar ({type(e).__name__}: {e})")

    a = np.asarray(a, dtype=float)
    if a.ndim == 2:
        # (member, frame) -> Ensemble-Mittel; (frame, 1) -> platt druecken
        a = a.ravel() if 1 in a.shape else a.mean(axis=0)
    if a.ndim != 1:
        raise Abbruch(EXIT_DATA, f"{p.name}: erwarte 1D oder 2D, bekam {a.ndim}D {a.shape}")
    return a


def lade_paar(pfad: str) -> tuple[np.ndarray, np.ndarray]:
    """(e_dft, e_ml) aus EINEM npz-Cache lesen.

    Spiegelt uq_mace.predictions.load_energies: verlangt 'e_dft' und nimmt
    'e_mace', sonst das Mittel von 'energies' ueber Achse 0 (Member-Achse,
    Form (M, F)). Damit laeuft das Skript direkt auf den Projekt-Caches
    predictions_*.npz und mace_energies_*.npz.
    """
    p = Path(pfad)
    if not p.exists():
        raise Abbruch(EXIT_USAGE, f"Datei nicht gefunden: {p}")
    if p.suffix.lower() != ".npz":
        raise Abbruch(EXIT_USAGE,
                      f"{p.name}: der Einzeldatei-Modus braucht ein .npz mit "
                      f"'e_dft' und 'e_mace'/'energies'. Sonst zwei Dateien "
                      f"angeben.")
    try:
        d = np.load(p, allow_pickle=False)
    except Exception as e:                                   # noqa: BLE001
        raise Abbruch(EXIT_USAGE, f"{p.name}: nicht lesbar ({type(e).__name__}: {e})")
    if "e_dft" not in d.files:
        raise Abbruch(EXIT_USAGE,
                      f"{p.name}: kein 'e_dft' enthalten (Keys: {', '.join(d.files)})")
    e_dft = np.asarray(d["e_dft"], dtype=float)
    if "e_mace" in d.files:
        e_ml = np.asarray(d["e_mace"], dtype=float)
    elif "energies" in d.files:
        e_ml = np.asarray(d["energies"], dtype=float).mean(axis=0)
    else:
        raise Abbruch(EXIT_USAGE,
                      f"{p.name}: 'e_dft' ohne 'e_mace'/'energies'")
    return e_dft.ravel(), e_ml.ravel()


def pruefe_daten(e_dft: np.ndarray, e_ml: np.ndarray, k_floor: int,
                 n_plan: int | None = None) -> np.ndarray:
    """Beide Arrays auf Vertraeglichkeit pruefen, dE zurueckgeben.

    n_plan: geplantes Kampagnenbudget (Live-Modus). Die Checkpoint-Existenz
    wird dann gegen n_plan geprueft, nicht gegen den aktuell vorliegenden
    Stand e_dft.size — sonst schlaegt die Pruefung bei noch wenigen Live-
    Punkten spuriell fehl, obwohl die Kampagne insgesamt genug Punkte plant.
    """
    if e_dft.size != e_ml.size:
        raise Abbruch(EXIT_DATA,
                      f"ungleiche Laenge: DFT hat {e_dft.size}, ML hat {e_ml.size} Werte. "
                      f"Beide Dateien muessen dieselben Frames in derselben "
                      f"Reihenfolge beschreiben.")
    if e_dft.size == 0:
        raise Abbruch(EXIT_DATA, "leere Eingabe")
    dE = e_dft - e_ml
    schlecht = ~np.isfinite(dE)
    if schlecht.any():
        raise Abbruch(EXIT_DATA,
                      f"{schlecht.sum()} nicht-endliche Werte in dE "
                      f"(Positionen {np.where(schlecht)[0][:5].tolist()}...)")
    grenze = dE.size if n_plan is None else n_plan
    ck = checkpoints_fuer(grenze, k_floor)
    if not ck:
        raise Abbruch(EXIT_DATA,
                      f"nur {grenze} Punkte (geplant) — im Raster liegt kein "
                      f"Checkpoint bei k >= {k_floor}. Mehr Punkte planen oder "
                      f"--k-floor senken (Referenzwert {K_FLOOR}).")
    if np.std(dE) == 0.0:
        raise Abbruch(EXIT_DATA, "dE ist konstant — die Modelle sind identisch?")
    return dE


# --------------------------------------------------------------------------
# Kerngrößen
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
    """SE(c) = c * sqrt((gamma2 + 2) / 4k), Delta-Methode fuer die
    Stichproben-Standardabweichung."""
    return float(c * np.sqrt(max(g2 + 2.0, 0.0) / (4.0 * max(k, 2))))


def cmax_gauss(R: float) -> float:
    """Schranke ohne Formkorrektur: N_eff/n = exp(-c^2) >= R."""
    return float(np.sqrt(-np.log(R)))


def cmax_skew(R: float, g1: float, g2: float, c_hi: float | None = None,
              warn: bool = True) -> float:
    """Kleinste positive Wurzel von c^2 - g1 c^3 + 7/12 g2 c^4 = -ln R,
    gedeckelt bei c_hi.

    Die Quartik wird ueber die Begleitmatrix vollstaendig faktorisiert
    (numpy.roots) statt mit einem Bracket-Verfahren: fuer gamma2 < 0 gibt es
    zwei positive Wurzeln, und ein Bracket liefert je nach Intervall die
    falsche. Existiert im Gueltigkeitsbereich keine Wurzel, wird c_hi
    zurueckgegeben — die OBERE Grenze, damit der Monitor nicht zu frueh feuert.
    """
    c_hi = C_VALID if c_hi is None else c_hi
    r = np.roots([(7.0 / 12.0) * g2, -g1, 1.0, 0.0, np.log(R)])
    pos = [x.real for x in r
           if abs(x.imag) <= 1e-8 * max(1.0, abs(x.real)) and 1e-12 < x.real < c_hi]
    if not pos:
        if warn:
            print(f"    [Hinweis] cmax_skew: keine Wurzel unterhalb C_VALID fuer "
                  f"g1={g1:+.3f}, g2={g2:+.3f} -> konservativer Rueckfall {c_hi:.3f}")
        return float(c_hi)
    return float(min(pos))


def log_neff_ratio(c: float, g1: float = 0.0, g2: float = 0.0) -> float:
    """log(N_eff/n) nach der Kumulantenentwicklung: -c^2 + g1 c^3 - 7/12 g2 c^4.

    g1 = g2 = 0 liefert den reinen Gauss-Praediktor. Der Vergleich mit dem
    exakten Kish-Wert ist die Gegenprobe: bei ensemble_L2c erklaert die
    Entwicklung die gemessene Abweichung von -1.50 % mit -1.48 % vollstaendig.
    """
    return float(-c ** 2 + g1 * c ** 3 - (7.0 / 12.0) * g2 * c ** 4)


def diagnose(R: float, g1: float, g2: float, rem_tol: float = R5_TOL) -> list[str]:
    """Voraussetzungen der Quartik pruefen -- einmal pro Modell, nicht pro Aufruf.

    Leere Liste = alles in Ordnung. Geprueft wird:

    * **Eindeutigkeit.** f'(c) = c [(7/3) g2 c^2 - 3 g1 c + 2]; die Klammer hat
      keine positive Nullstelle, wenn g1 <= 0 <= g2 oder g1^2 < (56/27) g2. Dann
      ist f streng monoton auf (0, inf) und die Wurzel eindeutig. Sonst kann es
      weitere geben -- cmax_skew nimmt die kleinste, was richtig ist, aber die
      Situation sollte sichtbar sein.
    * **N_eff <= n.** Cauchy-Schwarz erzwingt log(N_eff/n) <= 0. Geprueft wird
      das direkt an c_max ueber log_neff_ratio -- nicht ueber die Nullstellen
      von dessen Ableitung, denn deren ungueltige Zone ist bei g2 > 0 ein
      BEGRENZTES Intervall zwischen zwei Nullstellen: jenseits der groesseren
      ist der Wert wieder gueltig. Ein Test nur gegen die kleinere Nullstelle
      (frueherer Ansatz) erzeugt dort falsch-positive Meldungen.
    * **A2-Restglied.** (2 c_max)^5 / 5! <= rem_tol.

    Die Pruefungen sitzen an c_max, nicht am gemessenen c -- gefragt ist, ob die
    SCHRANKE tragfaehig ist, nicht ob die Reihe am aktuellen Arbeitspunkt haelt.
    """
    out: list[str] = []
    c = cmax_skew(R, g1, g2, warn=False)

    if not (g1 <= 0.0 <= g2 or (g2 > 0.0 and g1 ** 2 < (56 / 27) * g2)):
        out.append(f"f nicht streng monoton (g1^2={g1**2:.4g} vs "
                   f"(56/27) g2={(56/27)*g2:.4g}) - weitere positive Wurzeln "
                   f"moeglich; kleinste gewaehlt")

    lnr = log_neff_ratio(c, g1, g2)
    if lnr > 0.0:
        out.append(f"c_max={c:.4f} verletzt N_eff <= n (log(N_eff/n)={lnr:.4g} > 0, "
                   f"Cauchy-Schwarz) - der Wert ist dort bedeutungslos")

    rem = (2.0 * c) ** 5 / 120.0
    if rem > rem_tol:
        out.append(f"A2-Restglied (2c)^5/5!={rem:.3g} > {rem_tol} - Reihe bei "
                   f"c_max nicht mehr belastbar")
    return out


def neff_ratio(w: np.ndarray) -> float:
    """Exaktes Kish N_eff/n — annahmefrei, ohne Reihenentwicklung."""
    s2 = float(np.sum(w ** 2))
    return float(w.sum() ** 2 / s2 / w.size) if s2 > 0 else float("nan")


def gewichte(dE: np.ndarray, beta: float) -> np.ndarray:
    """w = exp(-beta dE), stabilisiert.

    Abzug des Minimums macht den groessten Exponenten exakt null: Ueberlauf ist
    damit ausgeschlossen, moeglich bleibt nur Unterlauf der ohnehin
    vernachlaessigbaren Gewichte. N_eff ist gegen einen konstanten Offset in dE
    invariant, der Eingriff aendert das Ergebnis also nicht.
    """
    return np.exp(-beta * (dE - dE.min()))


def _gpd_khat(x: np.ndarray) -> float:
    """Zhang & Stephens (2009), Posterior-Mittel des GPD-Formparameters."""
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
    """Pareto-Tail-Index der Gewichte. E[w^2] < unendlich  <=>  khat < 0.5.

    Achtung: stark verrauschter Schaetzer, SE faellt nur wie n^(-1/4). Bei
    wenigen hundert Punkten kann ein einzelner Wert die 0.5-Schwelle nicht
    entscheiden — deshalb wird khat hier auf dem VOLLEN Satz gerechnet, nie
    auf einem Praefix.
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
# Sequenzieller Monitor
# --------------------------------------------------------------------------
def checkpoint_grid(n: int, first_frac: float = FIRST_FRAC,
                    ratio: float = 1.4) -> np.ndarray:
    """Geometrisches Checkpoint-Raster ab first_frac*n bis n.

    Faustregel: der erste Blick bei 10 % der ohnehin geplanten Punkte. Das ist
    skalenfrei (n=500 -> ab 50, n=5000 -> ab 500) und liefert in beiden Faellen
    rund acht Blicke. Frueher zu schauen bringt wenig: unterhalb davon ist nicht
    nur SE(c) gross, sondern auch die Schranke selbst unzuverlaessig, und die
    Ersparnis waere ohnehin schon bei 90 %.

    ACHTUNG, zwei getrennte Groessen. Der Rasteranfang first_frac*n und der
    Filter K_FLOOR sind NICHT dasselbe. Fuer n < 500 liegt der Rasteranfang
    unter K_FLOOR und der Filter greift; fuer n >= 500 liegt er darueber und der
    Filter ist wirkungslos. Wer beides zu einem max(K_FLOOR, n//10) verschmilzt,
    bekommt fuer n=400 das Raster [50, 70, 98, ...] statt [56, 78, 109, ...] --
    also einen ERSTEN BLICK, der frueher liegt als validiert.
    """
    k0 = max(int(round(first_frac * n)), 10)
    ks = [k0]
    while ks[-1] * ratio < n:
        ks.append(int(round(ks[-1] * ratio)))
    ks.append(int(n))
    return np.array(sorted(set(ks)))


def checkpoints_fuer(n: int, k_floor: int = K_FLOOR,
                     first_frac: float = FIRST_FRAC) -> list[int]:
    """Das Raster, gefiltert auf k >= k_floor -- so wie monitor_split es tut."""
    ck = checkpoint_grid(n, first_frac)
    return [int(k) for k in ck if k_floor <= k <= n]


def se_cmax_boot(dE_prefix: np.ndarray, R: float, beta: float,
                 B: int, rng: np.random.Generator) -> float:
    """SE(c_max) aus B Bootstrap-Resamples DERSELBEN k Punkte.

    Analytisch geht es nicht: die Delta-Methode ueber dc_max/dgamma1 = c^3/f'
    hat f'(c_max) im Nenner, und der geht bei verrauschtem gamma1 gegen null —
    gemessen im Mittel 60 % zu hoch, mit Ausreissern bis Faktor 35.
    """
    k = dE_prefix.size
    idx = rng.integers(0, k, (B, k))
    proben = dE_prefix[idx]
    werte = np.empty(B)
    for i in range(B):
        _, g1, g2 = momente(proben[i], beta)
        werte[i] = cmax_skew(R, g1, g2, warn=False)   # im Hot Path stumm
    return float(werte.std())


def monitor_schritt(prefix: np.ndarray, R: float, beta: float, band: float,
                    B: int, rng: np.random.Generator) -> dict:
    """Ein einzelner Checkpoint-Schritt: alle Kenngroessen bei k = prefix.size.

    Regel:  c(k) - band*SE(c)  >  c_max(k) + band*SE(c_max)   =>  FAIL

    Von monitor() ausgelagert, damit derselbe Schritt auch einzeln (Live-
    Modus, ein Aufruf = ein Checkpoint) ausgefuehrt werden kann, ohne die
    Historie erneut durchzuwalzen.
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
    """Sequenzieller FAIL-only-Monitor ueber das Checkpoint-Raster.

    Einseitig: ein frueher PASS spart nichts, die Gewichte werden am Ende
    ohnehin vollstaendig gebraucht. Nur ein frueher FAIL spart Rechenzeit.
    Alle Groessen bei k benutzen ausschliesslich dE[:k] — der Monitor sieht
    die Zukunft nicht.

    Einzelsequenz-Fassung von uq_mace.screening.monitor_split. Zwei bewusste
    Unterschiede zur Bibliothek, beide ohne Wirkung auf die Entscheidung:
    dort laufen viele Sequenzen zeilenweise parallel, und die Schranke kommt
    aus cmax_skew_vec (Newton) statt aus cmax_skew (numpy.roots) — auf einem
    Gitter aus 3721 Parameterpaaren stimmen beide auf 4.7e-15 ueberein.

    Retrospektive Simulation ueber ein bereits fertiges dE: fuer den
    Live-Einzelschritt an einem wachsenden dE siehe monitor_schritt().
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
# Ausgabe
# --------------------------------------------------------------------------
def _bericht_live(erg: dict) -> str:
    """Kompakter Bericht fuer einen einzelnen Live-Checkpoint (siehe _live_schritt).

    Zeigt bewusst nur, was bei diesem k tatsaechlich berechnet wurde -- kein
    khat, kein exaktes N_eff/n, keine Restglied-Diagnose. Die volle
    Zertifizierung kommt erst mit dem Aufruf bei n >= n_plan (bericht()).
    """
    g = erg["gesamt"]
    z = []
    z.append("=" * 62)
    z.append("  KISH-SCREENING — Live-Checkpoint")
    z.append("=" * 62)
    z.append(f"  Punkte              {g['n']} von {g['n_plan']} geplant")
    z.append(f"  Temperatur          {g['T']:.1f} K   (beta = {g['beta']:.3f} 1/eV)")
    z.append(f"  Ziel R              {g['R']}")
    schritte = erg["monitor"]["schritte"] if erg.get("monitor") else []
    if schritte:
        s = schritte[0]
        z.append("")
        z.append(f"  c = beta*std(dE)    {s['c']:.4f}")
        z.append(f"  c_max (schief)      {s['c_max']:.4f}")
        z.append(f"  Abstand c - c_max   {s['abstand']:+.4f}   (Band {s['band']:.4f})")
    z.append("")
    z.append("=" * 62)
    z.append(f"  URTEIL: {erg['urteil']}")
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
    z.append("  KISH-SCREENING — traegt das Reweighting?")
    z.append("=" * 62)
    z.append(f"  Punkte n            {g['n']}")
    z.append(f"  Temperatur          {g['T']:.1f} K   (beta = {g['beta']:.3f} 1/eV)")
    z.append(f"  Ziel R              {g['R']}")
    z.append("")
    z.append(f"  std(dE)             {g['sigma']*1000:.3f} meV")
    z.append(f"  c = beta*std(dE)    {g['c']:.4f}")
    z.append(f"  Schiefe   gamma1    {g['gamma1']:+.4f}")
    z.append(f"  Kurtosis  gamma2    {g['gamma2']:+.4f}")
    z.append("")
    z.append(f"  c_max (Gauss)       {g['c_max_gauss']:.4f}")
    z.append(f"  c_max (schief)      {g['c_max']:.4f}   <- verwendet")
    z.append(f"  rho = c/c_max       {g['rho']:.3f}")
    z.append("")
    z.append(f"  N_eff/n (exakt)     {g['neff_ratio']:.4f}"
             f"   {'>=' if g['neff_ratio'] >= g['R'] else '<'} R = {g['R']}")
    z.append(f"  N_eff/n (Reihe)     {g['neff_ratio_reihe']:.4f}"
             f"   Gauss allein: {g['neff_ratio_gauss']:.4f}")
    z.append(f"  khat (Tail-Index)   {g['khat']:+.3f}"
             f"   {'Gate bestanden' if g['khat'] < KHAT_GATE else 'GATE VERLETZT'}")
    z.append(f"  Restglied (2c)^5/5! {g['r5']:.4f}"
             f"   {'ok' if g['r5'] <= R5_TOL else 'Reihe unbrauchbar'}")
    for h in erg.get("hinweise", []):
        z.append(f"  [Hinweis] {h}")

    m = erg.get("monitor")
    if m is not None:
        z.append("")
        z.append("-" * 62)
        z.append("  SEQUENZIELLER MONITOR")
        z.append("-" * 62)
        z.append(f"  Regel: c(k) - {g['band']:g}*SE(c) > c_max(k) + "
                 f"{g['band']:g}*SE(c_max)")
        z.append(f"  Raster: ab {g['first_frac']:.0%} von n, Faktor 1.4, "
                 f"gefiltert auf k >= {m['k_floor']}")
        z.append(f"  Checkpoints: {m['checkpoints']}")
        if zeige_schritte:
            z.append("")
            z.append(f"  {'k':>7}{'c':>9}{'c_max':>9}{'Abstand':>10}"
                     f"{'Band':>9}{'Urteil':>10}")
            z.append("  " + "-" * 52)
            for s in m["schritte"]:
                z.append(f"  {s['k']:>7}{s['c']:>9.4f}{s['c_max']:>9.4f}"
                         f"{s['abstand']:>+10.4f}{s['band']:>9.4f}"
                         f"{'FAIL' if s['feuert'] else 'weiter':>10}")
        z.append("")
        if m["gefeuert"]:
            z.append(f"  -> Abbruch bei k = {m['k_stop']} von {g['n']}")
            z.append(f"     {m['gespart']*100:.0f} % der DFT-Punkte waeren gespart worden.")
        else:
            z.append("  -> kein Abbruch. Der Monitor haette den Lauf durchlaufen lassen.")

    z.append("")
    z.append("=" * 62)
    z.append(f"  URTEIL: {erg['urteil']}")
    if erg.get("begruendung"):
        z.append(f"  {erg['begruendung']}")
    z.append("=" * 62)
    return "\n".join(z)


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------
def parser_bauen() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kish_screening.py",
        description="Prueft, ob thermodynamisches Reweighting statistisch traegt, "
                    "und simuliert den sequenziellen Abbruch-Monitor.",
        epilog="Exit-Codes: 0 PASS | 1 FAIL | 2 Aufruf | 3 Daten | 4 nicht belastbar",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dft", metavar="DFT_DATEI",
                   help="Referenzenergien (DFT). Enthaelt die Datei sowohl "
                        "'e_dft' als auch 'e_mace'/'energies' (Projekt-Cache), "
                        "kann ML_DATEI entfallen.")
    p.add_argument("ml", metavar="ML_DATEI", nargs="?",
                   help="Modellenergien (MACE o.ae.)")
    p.add_argument("-R", "--target", type=float, default=0.8, metavar="F",
                   help="gefordertes N_eff/n, in (0,1) (Default 0.8)")
    p.add_argument("-T", "--temperature", type=float, default=292.0, metavar="K",
                   help="Temperatur in Kelvin (Default 292)")
    p.add_argument("-u", "--units", default="eV", choices=sorted(UNITS),
                   help="Einheit der Eingabeenergien (Default eV)")
    p.add_argument("--key-dft", metavar="NAME", help="npz-Schluessel der DFT-Datei")
    p.add_argument("--key-ml", metavar="NAME", help="npz-Schluessel der ML-Datei")
    p.add_argument("-k", "--k-floor", type=int, default=K_FLOOR, metavar="N",
                   help=f"Checkpoints unterhalb k werden verworfen "
                        f"(Default {K_FLOOR})")
    p.add_argument("-N", "--n-plan", type=int, default=None, metavar="N",
                   help="geplantes Gesamtbudget der Kampagne. Ohne diese "
                        "Option laeuft das Checkpoint-Raster relativ zur "
                        "aktuell vorliegenden Punktzahl (Batch-Modus). Mit "
                        "--live noetig: legt das Raster fest, unabhaengig "
                        "davon, wie viele Punkte gerade vorliegen.")
    p.add_argument("--live", action="store_true",
                   help="Live-Checkpoint-Modus fuer die Einbettung in eine "
                        "laufende Kampagne: prueft nur den EINEN faelligen "
                        "Checkpoint bei k = aktuelle Punktzahl (kein "
                        "Raster-Walk), statt die Historie erneut zu simulieren. "
                        "Erfordert --n-plan. Unterhalb --k-floor: WEITER ohne "
                        "Check. Bei aktueller Punktzahl >= --n-plan: volle "
                        "Zertifizierung wie im Batch-Modus.")
    p.add_argument("--first-frac", type=float, default=FIRST_FRAC, metavar="F",
                   help=f"Rasteranfang als Anteil von n (Default {FIRST_FRAC}). "
                        f"Nicht mit --k-floor verwechseln: das Raster beginnt "
                        f"bei first_frac*n, k_floor schneidet danach ab.")
    p.add_argument("-b", "--band", type=float, default=1.0, metavar="F",
                   help="Bandbreite in Standardfehlern je Seite (Default 1.0)")
    p.add_argument("-B", "--bootstrap", type=int, default=200, metavar="N",
                   help="Resamples je Checkpoint fuer SE(c_max) (Default 200)")
    p.add_argument("--seed", type=int, default=0, metavar="N",
                   help="Zufallsstartwert, fuer reproduzierbare Laeufe (Default 0)")
    p.add_argument("--no-monitor", action="store_true",
                   help="nur die Kennzahlen, keine Checkpoint-Simulation")
    p.add_argument("--steps", action="store_true",
                   help="Tabelle aller Checkpoints ausgeben")
    p.add_argument("--json", action="store_true",
                   help="Ergebnis als JSON auf stdout (fuer Weiterverarbeitung)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="nur das Urteil (PASS/FAIL/UNKLAR)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _live_schritt(dE: np.ndarray, args, beta: float, k_floor: int) -> tuple[dict, int]:
    """Ein Live-Checkpoint: genau EIN Aufruf von monitor_schritt() bei k = dE.size.

    Kein Raster-Walk, kein khat-Gate, keine Restglied-/C_VALID-Diagnose --
    die sind laut analyses/13_sequential_screening/README.md ohnehin nur eine
    Nachpruefung auf dem PASS-Zweig bzw. fuer die finale Zertifizierung
    gedacht (siehe rechnen(), Zweig n >= n_plan). cmax_skew's c_hi-Fallback
    haelt die FAIL-Regel auch ausserhalb des Reihen-Gueltigkeitsbereichs
    konservativ.
    """
    n = dE.size
    gesamt = {"n": n, "n_plan": args.n_plan, "T": args.temperature, "beta": beta,
              "R": args.target, "units": args.units, "band": args.band,
              "k_floor": k_floor, "live": True}
    erg = {"gesamt": gesamt, "version": __version__, "hinweise": [], "warnungen": []}

    if n < k_floor:
        erg["urteil"] = "WEITER"
        erg["begruendung"] = (f"{n} von {args.n_plan} Punkten -- noch unter "
                              f"--k-floor={k_floor}, kein Check.")
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
        erg["begruendung"] = (f"Monitor feuert bei k={n}: c={schritt['c']:.4f} "
                              f"- {args.band:g}*SE > c_max={schritt['c_max']:.4f} "
                              f"+ {args.band:g}*SE")
        return erg, EXIT_FAIL

    erg["urteil"] = "WEITER"
    erg["begruendung"] = (f"{n} von {args.n_plan} Punkten, kein FAIL "
                          f"(Abstand {schritt['abstand']:+.4f}) -- Kampagne fortsetzen.")
    return erg, EXIT_PASS


def rechnen(args) -> tuple[dict, int]:
    if not 0.0 < args.target < 1.0:
        raise Abbruch(EXIT_USAGE, f"R muss in (0,1) liegen, bekam {args.target}")
    if args.temperature <= 0:
        raise Abbruch(EXIT_USAGE, f"Temperatur muss positiv sein, bekam {args.temperature}")
    if args.band < 0:
        raise Abbruch(EXIT_USAGE, f"Bandbreite darf nicht negativ sein, bekam {args.band}")
    if args.bootstrap < 20:
        raise Abbruch(EXIT_USAGE, f"--bootstrap sollte >= 20 sein, bekam {args.bootstrap}")

    faktor = UNITS[args.units]
    if args.ml is None:
        e_dft, e_ml = lade_paar(args.dft)
        e_dft, e_ml = e_dft * faktor, e_ml * faktor
    else:
        e_dft = lade_energien(args.dft, args.key_dft) * faktor
        e_ml = lade_energien(args.ml, args.key_ml) * faktor

    k_floor = args.k_floor
    if k_floor < 5:
        raise Abbruch(EXIT_USAGE, f"--k-floor muss >= 5 sein, bekam {k_floor}")
    if not 0.0 < args.first_frac <= 1.0:
        raise Abbruch(EXIT_USAGE,
                      f"--first-frac muss in (0,1] liegen, bekam {args.first_frac}")
    if args.live and args.n_plan is None:
        raise Abbruch(EXIT_USAGE, "--live erfordert --n-plan")
    if args.n_plan is not None and args.n_plan < k_floor:
        raise Abbruch(EXIT_USAGE,
                      f"--n-plan muss >= --k-floor sein, bekam {args.n_plan} < {k_floor}")

    # n_plan nur im Live-Modus an die Validierung durchreichen: der Batch-Pfad
    # baut den Monitor weiterhin ueber die tatsaechliche dE.size (siehe unten),
    # ein n_plan waere dort ein Fenster ohne Wirkung, das die "zu wenige
    # Punkte"-Pruefung nur lautlos aushebeln wuerde.
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

    # --- Belastbarkeit zuerst: ohne sie hat weder PASS noch FAIL Bedeutung ---
    warnungen = []
    if not np.isnan(kh) and kh >= KHAT_GATE:
        warnungen.append(
            f"khat = {kh:.3f} >= {KHAT_GATE}: E[w^2] existiert nicht, N_eff hat "
            f"keinen Populationsgrenzwert. Der berechnete Wert ist eine "
            f"Stichprobenzahl ohne Ziel.")
    if c >= C_VALID:
        warnungen.append(
            f"c = {c:.4f} >= C_VALID = {C_VALID:.4f}: die Kumulantenreihe ist hier "
            f"wertlos, c_max ist nicht bestimmbar.")
    elif r5 > R5_TOL:
        warnungen.append(
            f"Restglied (2c)^5/5! = {r5:.4f} > {R5_TOL}: die Reihe traegt bei "
            f"diesem c nicht mehr zuverlaessig.")
    # Voraussetzungen der Quartik -- an c_max, nicht am gemessenen c.
    # Die Nichtmonotonie ist ein HINWEIS, kein Fehler: cmax_skew nimmt die
    # kleinste positive Wurzel, und das ist die richtige. Sichtbar soll sie
    # trotzdem sein. Die anderen beiden Meldungen entwerten die Schranke.
    hinweise = [m for m in gesamt["diagnose"] if "monoton" in m]
    warnungen.extend(f"Schranke: {m}" for m in gesamt["diagnose"]
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

    # --- Urteil ---------------------------------------------------------
    exakt_fail = gesamt["neff_ratio"] < args.target
    monitor_fail = erg.get("monitor", {}).get("gefeuert", False)

    if warnungen and not exakt_fail:
        erg["urteil"] = "UNKLAR"
        erg["begruendung"] = warnungen[0]
        return erg, EXIT_UNRELIABLE
    if exakt_fail or monitor_fail:
        erg["urteil"] = "FAIL"
        teile = []
        if exakt_fail:
            teile.append(f"N_eff/n = {gesamt['neff_ratio']:.4f} < R = {args.target}")
        if monitor_fail:
            teile.append(f"Monitor feuert bei k = {erg['monitor']['k_stop']}")
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
        print(f"FEHLER: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("abgebrochen", file=sys.stderr)
        return 130

    if args.json:
        print(json.dumps(erg, indent=2, ensure_ascii=False, default=float))
    elif args.quiet:
        print(erg["urteil"])
    else:
        print(bericht(erg, args.steps))
    for w in erg.get("warnungen", []):
        print(f"WARNUNG: {w}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())