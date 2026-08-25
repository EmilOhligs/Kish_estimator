"""Tests fuer den Live-Checkpoint-Modus von kish_screening.py (-N/--n-plan, --live).

kish_screening.py liegt am Repo-Root, nicht im installierten Paket -- daher
wird ROOT hier von Hand auf sys.path gesetzt (gleiches Muster wie
tests/test_cache_konsistenz.py fuer den Cache-Pfad).

Hintergrund: der Live-Modus soll bei jedem neuen DFT-Batch waehrend einer
laufenden MD-Simulation erneut aufgerufen werden koennen, mit den bis dahin
gesammelten Punkten -- ohne dass sich das Checkpoint-Raster mit jedem Aufruf
verschiebt und ohne die bereits entschiedene Historie neu zu rechnen (siehe
README_kish_screening.md, Abschnitt 7a).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kish_screening as ks  # noqa: E402

R0 = 0.8
BETA = 1.0 / (8.617333262e-5 * 292)      # 1/eV bei 292 K, wie test_screening.py


def _schreibe_paar(tmp_path: Path, dE: np.ndarray) -> tuple[Path, Path]:
    """e_dft = dE, e_ml = 0 -- dE liegt damit exakt fest."""
    p_dft = tmp_path / "e_dft.npy"
    p_ml = tmp_path / "e_ml.npy"
    np.save(p_dft, dE.astype(float))
    np.save(p_ml, np.zeros_like(dE, dtype=float))
    return p_dft, p_ml


# ---------------------------------------------------------------------------
# monitor_schritt() -- Refactor-Sicherheit gegen monitor()
# ---------------------------------------------------------------------------
def test_monitor_schritt_reproduziert_ersten_schritt_von_monitor():
    rng_daten = np.random.default_rng(0)
    dE = rng_daten.normal(0.0, 0.01, 600)

    m = ks.monitor(dE, R0, BETA, ks.K_FLOOR, 1.0, 50,
                   np.random.default_rng(123))
    erster = m["schritte"][0]

    schritt = ks.monitor_schritt(dE[:erster["k"]], R0, BETA, 1.0, 50,
                                 np.random.default_rng(123))
    assert schritt == erster


# ---------------------------------------------------------------------------
# pruefe_daten() -- n_plan statt aktuellem Stand fuer die Checkpoint-Pruefung
# ---------------------------------------------------------------------------
def test_pruefe_daten_erlaubt_kleinen_live_stand_mit_n_plan():
    e_dft = np.zeros(20)
    e_ml = np.linspace(-0.01, 0.01, 20)
    dE = ks.pruefe_daten(e_dft, e_ml, ks.K_FLOOR, n_plan=5000)
    assert dE.size == 20


def test_pruefe_daten_ohne_n_plan_scheitert_bei_kleinem_satz():
    e_dft = np.zeros(20)
    e_ml = np.linspace(-0.01, 0.01, 20)
    with pytest.raises(ks.Abbruch):
        ks.pruefe_daten(e_dft, e_ml, ks.K_FLOOR)


# ---------------------------------------------------------------------------
# CLI-Validierung
# ---------------------------------------------------------------------------
def test_live_ohne_n_plan_ist_aufrufsfehler(tmp_path):
    p_dft, p_ml = _schreibe_paar(tmp_path, np.linspace(-0.01, 0.01, 80))
    args = ks.parser_bauen().parse_args([str(p_dft), str(p_ml), "--live"])
    with pytest.raises(ks.Abbruch) as exc:
        ks.rechnen(args)
    assert exc.value.code == ks.EXIT_USAGE


def test_n_plan_unter_k_floor_ist_aufrufsfehler(tmp_path):
    p_dft, p_ml = _schreibe_paar(tmp_path, np.linspace(-0.01, 0.01, 80))
    args = ks.parser_bauen().parse_args(
        [str(p_dft), str(p_ml), "--live", "-N", "10"])
    with pytest.raises(ks.Abbruch) as exc:
        ks.rechnen(args)
    assert exc.value.code == ks.EXIT_USAGE


# ---------------------------------------------------------------------------
# Live-Zweig: WEITER unterhalb k_floor, FAIL/WEITER am aktuellen Checkpoint,
# volle Zertifizierung sobald n_plan erreicht ist
# ---------------------------------------------------------------------------
def test_live_unter_k_floor_liefert_weiter_ohne_check(tmp_path):
    dE = np.linspace(-0.01, 0.01, 20)          # < K_FLOOR = 50
    p_dft, p_ml = _schreibe_paar(tmp_path, dE)
    args = ks.parser_bauen().parse_args(
        [str(p_dft), str(p_ml), "-N", "5000", "--live"])
    erg, code = ks.rechnen(args)
    assert code == ks.EXIT_PASS
    assert erg["urteil"] == "WEITER"
    assert erg["monitor"] is None
    assert erg["gesamt"]["n_plan"] == 5000


def test_live_erkennt_klares_fail_vor_n_plan(tmp_path):
    rng_daten = np.random.default_rng(1)
    dE = rng_daten.normal(0.0, 0.05, 120)      # c ~ 2.0, weit ueber c_max ~ 0.47
    p_dft, p_ml = _schreibe_paar(tmp_path, dE)
    args = ks.parser_bauen().parse_args(
        [str(p_dft), str(p_ml), "-N", "5000", "--live", "-B", "30"])
    erg, code = ks.rechnen(args)
    assert code == ks.EXIT_FAIL
    assert erg["urteil"] == "FAIL"
    assert erg["monitor"]["gefeuert"] is True
    assert len(erg["monitor"]["schritte"]) == 1
    assert "khat" not in erg["gesamt"]          # kein Gate mitten in der Kampagne


def test_live_laesst_klaren_nicht_fail_fall_weiterlaufen(tmp_path):
    rng_daten = np.random.default_rng(2)
    dE = rng_daten.normal(0.0, 0.001, 120)     # c ~ 0.04, weit unter c_max
    p_dft, p_ml = _schreibe_paar(tmp_path, dE)
    args = ks.parser_bauen().parse_args(
        [str(p_dft), str(p_ml), "-N", "5000", "--live", "-B", "30"])
    erg, code = ks.rechnen(args)
    assert code == ks.EXIT_PASS
    assert erg["urteil"] == "WEITER"
    assert erg["monitor"]["gefeuert"] is False


def test_live_faellt_bei_n_plan_erreicht_in_volle_zertifizierung(tmp_path):
    rng_daten = np.random.default_rng(3)
    dE = rng_daten.normal(0.0, 0.001, 100)
    p_dft, p_ml = _schreibe_paar(tmp_path, dE)

    args_live = ks.parser_bauen().parse_args(
        [str(p_dft), str(p_ml), "-N", "100", "--live", "-B", "30"])
    erg_live, code_live = ks.rechnen(args_live)

    args_batch = ks.parser_bauen().parse_args([str(p_dft), str(p_ml), "-B", "30"])
    erg_batch, code_batch = ks.rechnen(args_batch)

    assert not erg_live["gesamt"].get("live")
    assert "khat" in erg_live["gesamt"]
    assert erg_live["urteil"] == erg_batch["urteil"]
    assert code_live == code_batch
