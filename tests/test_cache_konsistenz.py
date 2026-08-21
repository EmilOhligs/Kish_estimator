"""Sehen alle Modelle wirklich dieselben Strukturen?

Der modelluebergreifende Vergleich setzt voraus, dass jede Cache-Datei denselben
Frame an derselben Position beschreibt. Das ist im Code nirgends abgesichert:

* Die L0-Subsamples entstehen durch **Slicing** des vollen Caches
  (`make_subsample.py`) -- das setzt voraus, dass Zeile j des vollen Caches
  Frame j der Quelldatei ist.
* Der L2-Subsample entsteht durch **Neurechnen** mit
  `compute_full_energies.py --indices` -- das setzt voraus, dass `--indices`
  dieselben Frames derselben Datei liest.

Zwei verschiedene Wege, die sich treffen muessen, ohne dass irgendwo verglichen
wird. Waeren sie um auch nur einen Frame verschoben, blieben alle Rechnungen
formal fehlerfrei und saehen plausibel aus -- die Modelle wuerden nur nicht mehr
dieselben Strukturen beschreiben, und jeder Modellvergleich waere hinfaellig.

**Der Trick, mit dem sich das trotzdem pruefen laesst:** `e_dft` ist
modellunabhaengig. Die DFT-Energie eines Frames haengt nicht davon ab, welches
MACE-Modell danebensteht. Beschreiben zwei Dateien dieselben Frames, muessen ihre
`e_dft`-Arrays **bitgleich** sein. `e_dft` wirkt damit als Fingerabdruck der
Frame-Auswahl.

Bitgleichheit ist dabei wesentlich: bei einer Verschiebung um einen Frame stuende
dort ein kleiner, aber von null verschiedener Wert.

Die Caches sind gitignored (unveroeffentlichte Forschungsdaten). Fehlen sie,
werden die Tests uebersprungen statt zu scheitern.
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

N_TOTAL = 14811          # Frames in results_all_corrected.xyz
N_SUB = 5000             # Groesse des modelluebergreifenden Subsamples

SUB_N5000 = {
    "mace-L0-01": "single_mace-L0-01_testfull_n5000.npz",
    "mace-L0-c-01": "single_mace-L0-c-01_testfull_n5000.npz",
    "mace-L2-c-01": "single_mace-L2-c-01_testfull_n5000.npz",
}
TESTBIG = {
    "ensemble_L2c": "mace_energies_ensemble_L2c_testbig.npz",
    "ensemble_L2c_pred": "predictions_ensemble_L2c_testbig.npz",
    "mace-L2-c-01": "single_mace-L2-c-01_testbig.npz",
    "mace-L2-c-02": "single_mace-L2-c-02_testbig.npz",
    "mace-L0-01": "single_mace-L0-01_testbig.npz",
    "mace-L0-c-01": "single_mace-L0-c-01_testbig.npz",
}
TESTFULL = {
    "mace-L0-01": "single_mace-L0-01_testfull.npz",
    "mace-L0-c-01": "single_mace-L0-c-01_testfull.npz",
}


def lade(datei):
    p = CACHE / datei
    if not p.exists():
        pytest.skip(f"{datei} nicht vorhanden (Cache ist gitignored)")
    return np.load(p, allow_pickle=True)


def e_dft_aller(gruppe):
    """{Modell: e_dft} fuer alle vorhandenen Dateien einer Gruppe."""
    out = {}
    for name, datei in gruppe.items():
        p = CACHE / datei
        if p.exists():
            out[name] = np.load(p, allow_pickle=True)["e_dft"]
    if len(out) < 2:
        pytest.skip("weniger als zwei Caches dieser Gruppe vorhanden")
    return out


# ---------------------------------------------------------------------------
# Der zentrale Test: e_dft als Fingerabdruck
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gruppe,label", [(SUB_N5000, "n5000"), (TESTBIG, "testbig"),
                                          (TESTFULL, "testfull")])
def test_alle_modelle_beschreiben_dieselben_frames(gruppe, label):
    """e_dft muss ueber alle Modelle einer Gruppe BITGLEICH sein.

    Faengt: verschobene Frame-Auswahl, vertauschte Reihenfolge, unterschiedliche
    Quelldateien, halb aktualisierte Caches.
    """
    werte = e_dft_aller(gruppe)
    ref_name, ref = next(iter(werte.items()))
    for name, e in werte.items():
        assert e.shape == ref.shape, (
            f"{label}: {name} hat {e.shape} Frames, {ref_name} hat {ref.shape}")
        assert np.array_equal(e, ref), (
            f"{label}: e_dft von {name} weicht von {ref_name} ab "
            f"(max |Diff| = {np.max(np.abs(e - ref)):.3e}) -- die Dateien "
            f"beschreiben NICHT dieselben Frames")


# ---------------------------------------------------------------------------
# Der Subsample selbst
# ---------------------------------------------------------------------------
def test_idx_ist_reproduzierbar():
    """Die Auswahl aus make_subsample.py muss sich exakt nachbauen lassen."""
    d = lade(SUB_N5000["mace-L0-01"])
    assert "idx" in d.files, "idx fehlt -- die Frame-Zuordnung waere nicht mehr belegbar"
    idx = d["idx"]
    erwartet = np.unique(np.linspace(0, N_TOTAL - 1, N_SUB).astype(int))
    assert np.array_equal(idx, erwartet), (
        "idx laesst sich nicht aus np.unique(np.linspace(0, N_TOTAL-1, n)) "
        "rekonstruieren -- entweder wurde N_TOTAL geaendert oder die Auswahl "
        "stammt aus einer anderen Fassung des Skripts")


def test_idx_ist_ueber_die_modelle_identisch():
    idxs = {}
    for name, datei in SUB_N5000.items():
        p = CACHE / datei
        if p.exists():
            d = np.load(p)
            assert "idx" in d.files, f"{name}: idx fehlt"
            idxs[name] = d["idx"]
    if len(idxs) < 2:
        pytest.skip("weniger als zwei n5000-Caches vorhanden")
    ref_name, ref = next(iter(idxs.items()))
    for name, i in idxs.items():
        assert np.array_equal(i, ref), f"idx von {name} weicht von {ref_name} ab"


def test_idx_eigenschaften():
    """Streng aufsteigend, im gueltigen Bereich, keine Dubletten."""
    idx = lade(SUB_N5000["mace-L0-01"])["idx"]
    assert idx.ndim == 1 and idx.size == N_SUB
    assert np.all(np.diff(idx) > 0), "idx ist nicht streng aufsteigend"
    assert idx.min() >= 0 and idx.max() < N_TOTAL
    assert np.unique(idx).size == idx.size


@pytest.mark.parametrize("modell", ["mace-L0-01", "mace-L0-c-01"])
def test_slicing_trifft_die_angegebenen_zeilen(modell):
    """Der Subsample muss exakt voll[idx] sein.

    Prueft den Slicing-Pfad aus make_subsample.py direkt: waere dort ein anderer
    Index-Vektor benutzt worden als der mitgeschriebene, faellt es hier auf.
    """
    if modell not in TESTFULL:
        pytest.skip("kein voller Cache fuer dieses Modell")
    voll = lade(TESTFULL[modell])
    sub = lade(SUB_N5000[modell])
    idx = sub["idx"]
    assert np.array_equal(voll["e_dft"][idx], sub["e_dft"])
    assert np.array_equal(voll["e_mace"][idx], sub["e_mace"])


def test_voller_cache_hat_die_erwartete_laenge():
    voll = lade(TESTFULL["mace-L0-01"])
    assert voll["e_dft"].size == N_TOTAL, (
        f"voller Cache hat {voll['e_dft'].size} statt {N_TOTAL} Frames -- "
        f"dann stimmt die Konstante in make_subsample.py nicht mehr")


# ---------------------------------------------------------------------------
# Formale Gesundheit der Caches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gruppe", [SUB_N5000, TESTBIG, TESTFULL])
def test_energien_sind_endlich_und_gleich_lang(gruppe):
    for name, datei in gruppe.items():
        p = CACHE / datei
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        e_dft = d["e_dft"]
        assert np.all(np.isfinite(e_dft)), f"{name}: nicht-endliche e_dft"
        if "e_mace" in d.files:
            e_mace = d["e_mace"]
            assert e_mace.shape == e_dft.shape, f"{name}: e_mace/e_dft ungleich lang"
            assert np.all(np.isfinite(e_mace)), f"{name}: nicht-endliche e_mace"
        if "energies" in d.files:
            en = d["energies"]
            assert en.ndim == 2, f"{name}: 'energies' sollte (member, frame) sein"
            assert en.shape[1] == e_dft.size, (
                f"{name}: 'energies' hat {en.shape} -- die zweite Achse muss die "
                f"Frame-Achse sein ({e_dft.size})")
            assert en.shape[0] < en.shape[1], (
                f"{name}: 'energies' hat {en.shape[0]} Member und {en.shape[1]} "
                f"Frames -- sind die Achsen vertauscht?")


def test_ensemble_mittel_stimmt_mit_e_mace():
    """Wo beide Schluessel vorliegen, muss e_mace das Mittel ueber die Member sein.

    Belegt zugleich, ueber welche Achse load_energies mitteln muss.
    """
    d = lade(TESTBIG["ensemble_L2c"])
    if "energies" not in d.files or "e_mace" not in d.files:
        pytest.skip("Datei hat nicht beide Schluessel")
    assert np.allclose(d["energies"].mean(axis=0), d["e_mace"], rtol=0, atol=1e-12)


def test_ausreisser_sind_im_n5000_satz_klar_abgesetzt():
    """Die vier DFT-Artefakte, die das Notebook bei dE > 0.1 eV entfernt.

    Geprueft wird nicht die Anzahl, sondern dass der Schnitt eine echte Luecke
    trifft -- sonst waere die Schwelle willkuerlich.
    """
    d = lade(SUB_N5000["mace-L2-c-01"])
    dE = d["e_dft"] - d["e_mace"]
    oben = dE > 0.10
    if not oben.any():
        pytest.skip("keine Ausreisser oberhalb 0.1 eV in diesem Cache")
    unten_max = dE[~oben].max()
    oben_min = dE[oben].min()
    assert oben_min > 3 * unten_max, (
        f"Schnitt bei 0.1 eV trennt nicht klar: groesster verbleibender Wert "
        f"{unten_max:.4f}, kleinster entfernter {oben_min:.4f}")
    assert not np.any(dE < -0.10), (
        "es gibt Ausreisser nach UNTEN -- der einseitige Schnitt dE > 0.1 im "
        "Notebook wuerde sie uebersehen")
