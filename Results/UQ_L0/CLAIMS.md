# Claim-Register

Für jede tragende Aussage die Zelle, die sie belegt. Zweck: wer eine Zelle löscht, sieht,
welche Behauptung damit unbelegt wird. Alle drei inhaltlichen Fehler aus Review-Runde 3
sind entstanden, weil beim Streichen von Kapitel 7 Belege verschwanden, auf die andere
Kapitel sich stützten.

| # | Aussage | steht in | belegt durch |
|---|---|---|---|
| K1 | Die Gauß-Schranke ist erst ab γ₁ > γ₁\* ≈ +0,107 konservativ, nicht ab γ₁ > 0 | §2.6 | §2.6-Zelle (Nulldurchgang numerisch + Näherung 7/12·γ₂·c_max) |
| K2 | Die Gauß-Schranke ist für die einseitige Regel untauglich (21 % gegen 0,6 % Fehlalarm) | §2.6-Lesart, §8.5 | §8.2-Zelle, Sweep-Tabelle bei ρ=0,90 |
| K3 | Der Kleine-k-Bias von γ₁,γ₂ zeigt unter der **einseitigen** Regel in die unsichere Richtung | §6.2 | §8.3-Zelle (Orakel gegen kausal) |
| K4 | Kein Look-ahead: jede Entscheidungsgröße bei k nutzt nur dE[:k] | §7.1 | Werkzeugblock-Verifikation + Docstrings in `uq_mace.screening` |
| K5 | Die Regel erkennt die L0-Modelle 30/30 korrekt beim ersten Checkpoint | §7.2 | §7.2-Zelle |
| K6 | k̂ ≥ 0,5 in 3/10 L0-01-Läufen → N_eff ohne Populationsgrenzwert | §7.2, §3.2 | §7.2-Zelle, Spalte khat |
| K7 | Rechtsschiefe ist die gefährliche Seite, nicht die schützende | §8.2 | §8.2-Zelle, Spiegeltest |
| K8 | Später schauen schlägt ein strengeres q | §8.4 | §8.4-Zelle, Teil A und B |
| K9 | Fehlende Abdeckung macht die einseitige Regel nur konservativer | §3.2 | **unbelegt im Notebook** — gemessen, aber die Zelle wurde nie eingebaut (§9 offen) |

**Regel:** Wird eine Zelle der rechten Spalte gelöscht oder inhaltlich geändert, muss die
zugehörige Aussage geprüft werden. `check_notebook.py` I4 fängt veraltete Zahlen, aber
keine veralteten Begründungen.
