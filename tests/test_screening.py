"""Tests fuer uq_mace.screening -- Schwerpunkt Entscheidungsschranke c_max.

Unabhaengige Referenz ist nicht ein zweites Nullstellenverfahren, sondern die
*analytische Konstruktion*: zu vorgegebenem c* und g1 laesst sich g2 so waehlen,
dass f(c*) = 0 exakt gilt. Damit ist die richtige Antwort vorab bekannt.

Der zweite Schwerpunkt ist cmax_skew_vec (Newton, vektorisiert, im Hot Path von
monitor_boot). Das Notebook verifizierte es bisher an sieben Testpaaren; hier
laeuft der Vergleich ueber ein Gitter aus rund 14 000 Punkten, inklusive der
Faelle mit zwei Wurzeln unterhalb C_VALID, bei denen Newton auf die falsche
springen koennte.
"""

import numpy as np
import pytest

from uq_mace import screening as scr
from uq_mace.screening import (
    C_VALID, K_FLOOR, Q_ALPHA, R5_TOL, checkpoint_grid, cmax_gauss, cmax_skew,
    cmax_skew_vec,
    diagnose, log_neff_ratio, monitor_boot, monitor_split, se_c, stat_D,
)

R0 = 0.8
BETA = 1.0 / (8.617333262e-5 * 292)      # 1/eV bei 292 K


@pytest.fixture(autouse=True)
def _ctx():
    scr.configure(beta=BETA, R=R0)


def f_quartik(c, g1, g2, R):
    return c**2 - g1 * c**3 + (7 / 12) * g2 * c**4 + np.log(R)


def g1_fuer_wurzel(c_stern, g2, R):
    """g1 so waehlen, dass f(c*) = 0 exakt gilt.

    Umgekehrt (g2 aus c* und g1) waere ebenso moeglich, liefert aber fuer
    kleine c* absurde Kurtosis-Werte (g2 ~ 200); so bleiben alle erzeugten
    Parameter im realistischen Bereich.
    """
    return (c_stern**2 + (7 / 12) * g2 * c_stern**4 + np.log(R)) / c_stern**3


def wurzeln(g1, g2, R):
    r = np.roots([(7 / 12) * g2, -g1, 1.0, 0.0, np.log(R)])
    return sorted(x.real for x in r if abs(x.imag) < 1e-8 and x.real > 1e-12)


# ---------------------------------------------------------------------------
# Analytische Referenzen
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("R", [0.6, 0.7, 0.8, 0.9, 0.99])
def test_gauss_geschlossene_form(R):
    assert cmax_gauss(R) == pytest.approx(np.sqrt(-np.log(R)), rel=1e-14)
    assert cmax_skew(R, 0.0, 0.0) == pytest.approx(cmax_gauss(R), rel=1e-12)


@pytest.mark.parametrize("faktor", [0.85, 1.0, 1.15, 1.30])
@pytest.mark.parametrize("g2", [-0.2, 0.0, 0.39, 1.0])
@pytest.mark.parametrize("R", [0.7, 0.8, 0.9])
def test_konstruierte_wurzel_wird_getroffen(faktor, g2, R):
    """Statt eine Wurzel zu suchen, eine vorgeben -- die richtige Antwort ist
    damit vorab bekannt, ohne ein zweites Nullstellenverfahren als Referenz.

    c* wird relativ zu cmax_gauss(R) gesetzt: absolute Werte wuerden fuer
    grosses R eine Schiefe von 1.5 und mehr erfordern, und dann existiert eine
    FRUEHERE Wurzel -- c* waere nicht mehr die kleinste und der Vergleich
    unten pruefte etwas anderes als gemeint.
    """
    c_stern = faktor * cmax_gauss(R)
    g1 = g1_fuer_wurzel(c_stern, g2, R)
    assert f_quartik(c_stern, g1, g2, R) == pytest.approx(0.0, abs=1e-14)

    # c* muss auch die KLEINSTE positive Wurzel sein, sonst prueft der Vergleich
    # unten etwas anderes als gemeint.
    unten = np.linspace(1e-6, c_stern * (1 - 1e-9), 20000)
    assert np.all(f_quartik(unten, g1, g2, R) < 0.0)

    c = cmax_skew(R, g1, g2, warn=False)
    if c_stern < C_VALID:
        assert c == pytest.approx(c_stern, rel=1e-10)
    else:
        # Konstruierte Wurzel jenseits des Gueltigkeitsbereichs: cmax_skew darf
        # sie NICHT liefern, sondern muss auf C_VALID zurueckfallen. Das ist die
        # konservative Richtung -- eine hohe Schranke laesst den FAIL-only-
        # Monitor nicht feuern.
        assert c == pytest.approx(C_VALID)


@pytest.mark.parametrize("g1", [-0.8, -0.3, 0.0, 0.25, 0.501])
@pytest.mark.parametrize("g2", [0.0, 0.2, 0.39, 1.0])
@pytest.mark.parametrize("R", [0.7, 0.8, 0.9])
def test_rueckrechnung_reproduziert_R(g1, g2, R):
    c = cmax_skew(R, g1, g2, warn=False)
    if c < C_VALID - 1e-9:
        assert np.exp(log_neff_ratio(c, g1, g2)) == pytest.approx(R, abs=1e-12)
    else:
        # Rueckfall auf C_VALID: dann darf es im Gueltigkeitsbereich auch
        # wirklich keine Wurzel geben (sonst waere der Rueckfall ein Fehler).
        pruef = np.linspace(1e-6, C_VALID, 20000)
        assert np.all(f_quartik(pruef, g1, g2, R) < 0.0)


def test_gemessene_L2c_werte():
    """ensemble_L2c: gamma1 = +0.501, gamma2 = +0.390 (Kap. 2.6)."""
    assert cmax_skew(0.8, 0.501, 0.390) == pytest.approx(0.5285501477, abs=1e-9)
    assert diagnose(0.8, 0.501, 0.390) == []


# ---------------------------------------------------------------------------
# Konditionierung und Randverhalten
# ---------------------------------------------------------------------------
def test_winziges_g2_erzeugt_keine_scheinwurzel():
    """numpy.roots liefert bei g2 ~ 1e-16 eine Wurzel bei ~1e16. Sie darf nicht
    durchschlagen, weil ohnehin die kleinste positive genommen wird."""
    assert cmax_skew(R0, 0.501, 1e-16, warn=False) == pytest.approx(
        cmax_skew(R0, 0.501, 0.0, warn=False), rel=1e-10)


def test_rueckfall_auf_c_hi_statt_auf_gauss(capsys):
    """Kein Vorzeichenwechsel unterhalb C_VALID -> OBERE Grenze. Die Gauss-
    Schranke laege tiefer und liesse den FAIL-only-Monitor zu frueh feuern."""
    c = cmax_skew(R0, 3.0, 0.0)          # Schiefekorrektur haelt f unter null
    assert c == pytest.approx(C_VALID)
    assert "Rueckfall" in capsys.readouterr().out
    assert c > cmax_gauss(R0)


def test_warn_schaltbar(capsys):
    cmax_skew(R0, 3.0, 0.0, warn=False)
    assert capsys.readouterr().out == ""


def test_c_hi_deckelt():
    assert cmax_skew(R0, 0.501, 0.390, c_hi=0.4) == pytest.approx(0.4)


def test_C_VALID_folgt_aus_R5_TOL():
    assert (2 * C_VALID) ** 5 / 120.0 == pytest.approx(R5_TOL, rel=1e-12)


def test_monoton_fallend_in_R():
    cs = [cmax_skew(R, 0.501, 0.390, warn=False) for R in (0.6, 0.7, 0.8, 0.9, 0.95)]
    assert all(a > b for a, b in zip(cs, cs[1:]))


def test_rechtsschiefe_hebt_linksschiefe_senkt():
    assert cmax_skew(R0, +0.501, 0.0, warn=False) > cmax_gauss(R0)
    assert cmax_skew(R0, -0.501, 0.0, warn=False) < cmax_gauss(R0)


def test_negatives_g2_liefert_die_kleinere_wurzel():
    """Der Fall, an dem ein naives Bracket [0, 3] scheitert (real aufgetreten
    bei mace-L0-01, gamma2 = -0.23)."""
    g1, g2 = 0.501, -0.390
    w = wurzeln(g1, g2, R0)
    assert len(w) == 2
    assert cmax_skew(R0, g1, g2, warn=False) == pytest.approx(min(w), rel=1e-11)


@pytest.mark.parametrize("g1,g2", [(0.501, 0.390), (-0.501, 0.390),
                                   (0.501, -0.390), (-0.725, -3.88)])
def test_erster_vorzeichenwechsel_auf_dichtem_gitter(g1, g2):
    """Die einzige Pruefung, die numpy.roots gar nicht benutzt.

    f wird auf 200 001 Punkten ausgewertet und der erste Aufwaertsdurchgang
    gesucht. Der letzte Parametersatz hat ZWEI Wurzeln unterhalb C_VALID, nur
    0.03 auseinander -- genau die Konstellation, in der ein Bracket-Verfahren
    je nach Intervall die falsche liefert.
    """
    grid = np.linspace(1e-6, C_VALID, 200_001)
    h = grid[1] - grid[0]
    fv = f_quartik(grid, g1, g2, R0)
    up = np.where((fv[:-1] < 0) & (fv[1:] >= 0))[0]
    assert up.size >= 1, "kein Aufwaertsdurchgang im Gueltigkeitsbereich"
    assert cmax_skew(R0, g1, g2, warn=False) == pytest.approx(grid[up[0]], abs=2 * h)


# ---------------------------------------------------------------------------
# cmax_skew_vec -- Newton im Hot Path
# ---------------------------------------------------------------------------
def test_vec_stimmt_mit_skalar_auf_grossem_gitter():
    G1 = np.linspace(-1.0, 2.0, 61)
    G2 = np.linspace(-2.0, 3.0, 61)
    A, B = np.meshgrid(G1, G2, indexing="ij")
    N = cmax_skew_vec(A, B)
    E = np.array([[cmax_skew(R0, a, b, warn=False) for b in G2] for a in G1])
    assert np.abs(N - E).max() < 1e-6


def test_vec_trifft_die_kleinere_wurzel_bei_zwei_wurzeln_unter_C_VALID():
    """Newton startet bei cmax_gauss = 0.472 und koennte auf die groessere
    springen. Betroffen ist der Bereich g1 < 0, g2 stark negativ."""
    paare = [(g1, g2) for g1 in np.linspace(-3, 0, 61)
             for g2 in np.linspace(-12, -2, 61)
             if len([x for x in wurzeln(g1, g2, R0) if x < C_VALID]) >= 2]
    assert len(paare) > 50, "Testfaelle nicht gefunden - Bereich pruefen"
    g1s = np.array([p[0] for p in paare])
    g2s = np.array([p[1] for p in paare])
    klein = np.array([min(x for x in wurzeln(*p, R0) if x < C_VALID) for p in paare])
    assert np.abs(cmax_skew_vec(g1s, g2s) - klein).max() < 1e-6


def test_vec_faellt_auf_C_VALID_zurueck_wenn_keine_wurzel():
    assert cmax_skew_vec(np.array([3.0]), np.array([0.0]))[0] == pytest.approx(C_VALID)


def test_vec_skalar_broadcast():
    assert cmax_skew_vec(0.501, 0.390) == pytest.approx(
        cmax_skew(R0, 0.501, 0.390), rel=1e-9)


def test_vec_wirft_wenn_R_ausserhalb_des_gueltigkeitsbereichs():
    scr.configure(beta=BETA, R=0.55)      # cmax_gauss = 0.773 > C_VALID
    with pytest.raises(ValueError, match="Gueltigkeitsbereich"):
        cmax_skew_vec(np.array([0.5]), np.array([0.4]))


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------
def test_diagnose_meldet_nichtmonotonie():
    assert any("nicht streng monoton" in m for m in diagnose(R0, 0.501, -0.390))


def test_diagnose_meldet_A2_restglied():
    assert any("A2-Restglied" in m for m in diagnose(0.55, 0.0, 0.0))
    assert diagnose(0.9, 0.0, 0.0) == []


def test_diagnose_linksschiefe_ist_immer_monoton():
    for g1 in (-0.1, -1.0, -5.0):
        for g2 in (0.0, 0.39, 3.0):
            assert not any("monoton" in m for m in diagnose(R0, g1, g2))


def test_diagnose_folgt_rem_tol():
    assert diagnose(0.7, 0.501, 0.390, rem_tol=0.5) == []
    assert any("A2-Restglied" in m for m in diagnose(0.7, 0.501, 0.390, rem_tol=0.01))


# ---------------------------------------------------------------------------
# monitor_split -- die einfache Regel (Baender um c und c_max statt um D)
# ---------------------------------------------------------------------------
def _pool(c_ziel, n=20000, seed=7):
    """dE-Pool mit realistischer Schiefe, skaliert auf beta*std = c_ziel."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(4.0, 1.0, n)                        # g1 = +1.0, g2 = +1.5
    return (x - x.mean()) / x.std(ddof=1) * (c_ziel / BETA)


def _kish(x):
    """Exaktes N_eff/n, schrankenfrei -- die Wahrheit ohne Kumulantennaeherung."""
    w = np.exp(-BETA * (x - x.min()))
    return (w.sum()**2 / np.sum(w**2)) / x.size


@pytest.fixture
def seqs():
    """150 Sequenzen a 400, dE mit realistischer Schiefe, weit unter der Schranke."""
    rng = np.random.default_rng(7)
    pool = _pool(0.30)
    return pool[rng.integers(0, pool.size, (150, 400))]


@pytest.fixture
def seqs_grenzfall():
    """Sequenzen dicht an der Schranke -- nur hier entscheidet die Bandbreite
    ueberhaupt etwas. Die strukturellen Tests oben laufen weit davon entfernt
    und wuerden eine falsche Schwelle nicht bemerken."""
    rng = np.random.default_rng(11)
    g1, g2 = 1.0, 1.5                                 # Gamma(4)
    c_grenz = cmax_skew(R0, g1, g2, warn=False)
    pool = _pool(1.15 * c_grenz, seed=11)             # knapp ueber der Schranke
    return pool[rng.integers(0, pool.size, (200, 400))]


def test_split_default_q_ist_eins():
    """Ein Standardfehler je Seite. Bewusst NICHT Q_ALPHA -- bei zwei
    verglichenen Baendern ist q kein Niveau (siehe Docstring)."""
    import inspect
    assert inspect.signature(monitor_split).parameters["q"].default == 1.0
    assert inspect.signature(monitor_boot).parameters["q"].default == Q_ALPHA


@pytest.mark.parametrize("q", [0.8, 1.0, 1.64])
def test_split_feuert_nie_frueher_als_boot(seqs, q):
    """Strukturell, nicht empirisch: SE(c) + SE(c_max) >= SE(D) immer, weil
    die Summe zweier Streuungen ihre Quadratur nie unterschreitet. Gilt nur
    bei GLEICHEM q -- die Defaults der beiden Funktionen sind verschieden."""
    fb, kb = monitor_boot(seqs, q=q, rng=np.random.default_rng(3))
    fs, ks = monitor_split(seqs, q=q, rng=np.random.default_rng(3))
    assert np.all(fs <= fb), "monitor_split feuert, wo monitor_boot es nicht tut"
    both = fs & fb
    assert np.all(ks[both] >= kb[both]), "monitor_split feuert frueher als monitor_boot"


def test_split_rueckgabeform_wie_boot(seqs):
    fs, ks = monitor_split(seqs, rng=np.random.default_rng(3))
    assert fs.shape == (seqs.shape[0],) and fs.dtype == bool
    assert ks.shape == (seqs.shape[0],)
    assert np.all(ks[~fs] == -1)
    assert np.all(ks[fs] >= K_FLOOR)


@pytest.mark.parametrize("q", [0.6, 1.0, 1.4])
def test_split_entscheidung_gegen_unabhaengige_nachrechnung(seqs_grenzfall, q):
    """Die Entscheidung Zeile fuer Zeile gegen eine getrennt hingeschriebene
    Implementierung, an EINEM Checkpoint und mit demselben Zufallsstrom.

    Das ist der Test, der die Schwelle wirklich festnagelt: er vergleicht nicht
    zwei Umformungen derselben Formel, sondern das Ergebnis der Funktion gegen
    eine Rechnung, die c, c_max, SE(c) und SE(c_max) eigenstaendig bildet.
    Deshalb laeuft er auf Grenzfall-Daten -- weit von der Schranke waere jede
    Bandbreite gleichwertig und der Test wertlos.
    """
    k, B = 100, 150
    sub = seqs_grenzfall[:, :k]

    fired, kfire = monitor_split(sub, q=q, B=B, checkpoints=[k], k_floor=k,
                                 rng=np.random.default_rng(5))

    # unabhaengig nachgerechnet, gleicher Zufallsstrom
    rng = np.random.default_rng(5)
    idx = rng.integers(0, k, (sub.shape[0], B, k))
    boot = np.take_along_axis(sub[:, None, :], idx, axis=2).reshape(-1, k)
    Db, cb, _, _ = stat_D(boot)
    c_max_boot = (cb - Db).reshape(sub.shape[0], B)

    y = sub - sub.mean(1, keepdims=True)
    m2 = (y**2).mean(1); m3 = (y**3).mean(1); m4 = (y**4).mean(1)
    c_hat = BETA * np.sqrt(m2 * k / (k - 1))
    g1_hat = m3 / m2**1.5
    g2_hat = m4 / m2**2 - 3.0
    cmax_hat = np.array([cmax_skew(R0, a, b, warn=False)
                         for a, b in zip(g1_hat, g2_hat)])
    se_c_hat = c_hat * np.sqrt(np.maximum(g2_hat + 2.0, 0.0) / (4*k))
    erwartet = (c_hat - q*se_c_hat) > (cmax_hat + q*c_max_boot.std(1))

    assert erwartet.any() and not erwartet.all(), \
        "Testdaten liegen nicht im Grenzfall -- der Vergleich waere aussagelos"
    assert np.array_equal(fired, erwartet)
    assert np.all(kfire[fired] == k) and np.all(kfire[~fired] == -1)


def test_split_bandbreite_wirkt(seqs_grenzfall):
    """Ein breiteres Band muss weniger feuern -- sonst geht die Bandbreite gar
    nicht in die Entscheidung ein."""
    raten = [monitor_split(seqs_grenzfall, q=q, B=80,
                           rng=np.random.default_rng(3))[0].mean()
             for q in (0.4, 0.7, 1.0, 1.4, 2.0)]
    assert all(a >= b for a, b in zip(raten, raten[1:])), raten
    assert raten[0] - raten[-1] > 0.25, \
        f"Bandbreite aendert die Rate kaum ({raten}) -- Testfall zu leicht"


def test_split_beide_unsicherheiten_gehen_ein(seqs_grenzfall):
    """Die Rate muss zwischen den beiden einseitigen Varianten liegen: nur
    SE(c) waere zu schmal, doppeltes SE(c) trifft SE(c_max) nicht.

    Faengt die Fehler 'SE(c_max) weggelassen' und 'stattdessen 2*SE(c)'.
    """
    k, B = 100, 150
    sub = seqs_grenzfall[:, :k]
    D, c, g1, g2 = stat_D(sub)
    rng = np.random.default_rng(5)
    idx = rng.integers(0, k, (sub.shape[0], B, k))
    boot = np.take_along_axis(sub[:, None, :], idx, axis=2).reshape(-1, k)
    Db, cb, _, _ = stat_D(boot)
    se_cm = (cb - Db).reshape(sub.shape[0], B).std(1)
    se_cc = se_c(c, g2, k)

    echt = monitor_split(sub, B=B, checkpoints=[k], k_floor=k,
                         rng=np.random.default_rng(5))[0]
    nur_c = D > se_cc                       # SE(c_max) weggelassen
    zwei_c = D > 2*se_cc                    # 2*SE(c) statt SE(c)+SE(c_max)
    kein_band = D > 0.0                     # gar kein Band

    assert se_cm.min() > 0.0
    assert echt.sum() < nur_c.sum(), "SE(c_max) scheint nicht einzugehen"
    assert echt.sum() < kein_band.sum(), "es wird gar kein Band abgezogen"
    assert not np.array_equal(echt, zwei_c), \
        "2*SE(c) liefert dieselbe Entscheidung -- SE(c_max) waere ersetzbar"
    # Welche der beiden Streuungen groesser ist, haengt von der Verteilung ab
    # (bei den echten L2c-Daten SE(c_max) > SE(c), hier umgekehrt) -- daher wird
    # hier nur geprueft, DASS beide eingehen, nicht in welcher Rangfolge.


def test_split_urteilt_richtig_bei_bekannter_wahrheit():
    """Der Unit-Test-Zwilling zu §7.2: zwei Pools, deren exaktes Kish-N_eff/n
    weit ueber bzw. weit unter R liegt. Kein Fehlalarm auf dem guten, volle
    Erkennung auf dem schlechten."""
    rng = np.random.default_rng(23)
    gut = _pool(0.33, seed=1)                 # wie ensemble_L2c, rho ~ 0.6
    schlecht = _pool(1.40, seed=2)            # wie mace-L0-01,    rho ~ 2.5
    assert _kish(gut) > R0 + 0.10, _kish(gut)          # 0.923
    assert _kish(schlecht) < R0 - 0.30, _kish(schlecht)  # 0.471

    fg = monitor_split(gut[rng.integers(0, gut.size, (100, 500))],
                       B=80, rng=np.random.default_rng(43))[0]
    fs, ks = monitor_split(schlecht[rng.integers(0, schlecht.size, (100, 500))],
                           B=80, rng=np.random.default_rng(43))
    assert fg.mean() == 0.0, f"Fehlalarm auf dem guten Modell: {fg.mean():.1%}"
    assert fs.mean() == 1.0, f"nur {fs.mean():.1%} Erkennung auf dem schlechten"
    assert np.all(ks[fs] == checkpoint_grid(500)[0])   # erster Checkpoint


def test_split_grosses_q_feuert_nie(seqs):
    fs, ks = monitor_split(seqs, q=50.0, rng=np.random.default_rng(3))
    assert not fs.any() and np.all(ks == -1)


def test_split_respektiert_k_floor(seqs):
    fs, ks = monitor_split(seqs, q=0.0, k_floor=120, rng=np.random.default_rng(3))
    assert np.all(ks[fs] >= 120)


def test_split_deterministisch_bei_gleichem_seed(seqs):
    a = monitor_split(seqs, rng=np.random.default_rng(3))
    b = monitor_split(seqs, rng=np.random.default_rng(3))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
