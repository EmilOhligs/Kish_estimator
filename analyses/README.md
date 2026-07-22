# Analysen — Landkarte

Ein Ordner pro Thema. **Skript und seine Ergebnisse liegen zusammen**: jedes Skript
schreibt Plots und CSVs standardmäßig in sein eigenes Verzeichnis (`--outdir`
überschreibt das).

Der gemeinsame MACE-Cache liegt bewusst außerhalb, unter `cache/`, weil mehrere
Analysen dieselben Vorhersagen brauchen und MACE nur **einmal** je
(Ensemble, Testsatz) laufen soll.

Aufruf immer vom Projekt-Root, z. B.:

```bash
python analyses/04_khat_uncertainty/khat_uncertainty.py
```

---

## Reihenfolge der Auswertung

Die Nummerierung folgt der logischen Kette, nicht der Entstehungsreihenfolge.

| Ordner | Frage | Kernaussage |
|---|---|---|
| **01_weight_distribution** | Wie sind die echten w_i verteilt? | CV = 0.30, N_eff/n = 0.917, k̂ = 0.065 |
| **02_energy_difference_qq** | Ist ΔE gaußisch? | Nein (p = 0.0004), γ₁ = +0.50 — trotzdem Gauß-Prädiktor auf 1.4 % genau |
| **03_running_estimates** | Wie schnell konvergieren c, γ₁, N_eff, k̂? | c ab k≈14, γ₁ erst ab k≈210 |
| **04_khat_uncertainty** | Wie sicher ist k̂ — die Existenzbedingung? | k̂<0.5 mit 3.7 SE sicher, k̂<0.25 mit 1.6 SE **nicht** |
| **05_regime_study** | Ab welchem c bricht der Gauß-Prädiktor? | Gefahrenzone über c ≈ 0.5; Schiefe wirkt schützend |
| **06_neff_spectrum** | Wie verhalten sich 10 Verteilungen über das N_eff-Spektrum? | N_eff und k̂ sind unabhängig: gleiches N_eff, verschiedene Verlässlichkeit |
| **07_convergence** | Ist das Permutationsband ehrlich? | Nein — Endlichkeitskorrektur; Bootstrap statt Permutation |
| **08_synthetic_basics** | Grundverständnis von N_eff (synthetische Gewichte) | N_eff/n = 1/(1+CV²), verteilungsfrei |
| **09_ensemble_evaluation** | RMSE und σ(R)-vs-Fehler je Ensemble | Grundlage des DFT-freien Pfads |
| **10_training** | Neue MACE-Member trainieren | GPU-Job |
| **11_error_correlation** | Sind die Kraftfehler räumlich korreliert? | Zerfall auf Grundlinie ab ~2.5 Å; √N nicht widerlegt (Hochpassfilter, k→0 blind) |
| **12_screening** | Trägt das Modell — **vor** der MD? | c aus dem Testsatz, +8.7 % Ensemble-Korrektur, √N auf 128 Mol.: 0.804 vs. Paper 0.814. `validate_ensemble_shift.py` beweist die Korrektur an analytisch bekannter Wahrheit und zeigt die Abdeckungsgrenze |

---

## Was wo hingehört

**01–04 arbeiten auf echten Daten** (DFT + MACE aus `cache/`).
**05–08 sind Simulation/Theorie** und laufen ohne DFT.
**09–10 sind Infrastruktur** (Modellbewertung, Training).

Wer den Stand nachvollziehen will, liest in dieser Reihenfolge:
`01 → 02 → 04 → 03`, dann für die Theorie `05` und `06`.

---

## Gemeinsame Daten

```
cache/
  predictions_<ensemble>_test<testset>.npz   MACE-Energien UND Kräfte (get_predictions)
  mace_energies_<ensemble>_test<testset>.npz  ältere Variante, nur Energien
```

Beide werden von `uq_mace.predictions` gelesen (`load_energies`, `load_weights`).
Der Ordner ist **gitignored** — er enthält DFT-Energien aus `data/`.

Skripte, die reale Daten brauchen, nehmen den Cache per Default:

```bash
python analyses/03_running_estimates/running_c_and_skew.py
python analyses/04_khat_uncertainty/khat_uncertainty.py
python analyses/02_energy_difference_qq/qq_plot_energy_difference.py \
    --energies cache/mace_energies_ensemble_L2c_testbig.npz
```

---

## Zugehörige Notizen

Die inhaltliche Auswertung steht in `notebooks/` (nicht im Repo veröffentlicht):

- `map.md` — Gesamtworkflow, sequenzielle Entscheidungslogik
- `gauss_naeherung_gueltigkeit.md` — Kumulantenherleitung mit allen Annahmen
- `regime_studie_methodik.md` — warum c = β·std(ΔE) die richtige Achse ist
- `plan_konvergenz_simulation.md` — Versuchsplan zu 07
- `ensemble_korrektur.md` — Herleitung zu 12, mit numerischer Prüfung und offener Prämisse
