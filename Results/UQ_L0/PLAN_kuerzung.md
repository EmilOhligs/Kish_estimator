# Plan: Kapitel 7 entzerren und Text kürzen

Bestandsaufnahme des aktuellen Notebooks (102 Zellen):

| | Wörter Markdown | Wörter Code | Anteil MD |
|---|---|---|---|
| **gesamt** | 11 691 | 6 052 | 66 % |
| Kapitel 1 | 782 | 196 | 80 % |
| Kapitel 2 | 1 963 | 264 | 88 % |
| Kapitel 3 | 554 | **0** | 100 % |
| Kapitel 4 | 740 | 420 | 64 % |
| Kapitel 5 | 743 | 531 | 58 % |
| Kapitel 6 | 938 | 521 | 64 % |
| **Kapitel 7** | **5 449** | **3 596** | 60 % |

Kapitel 7 ist **47 % des gesamten Notebooks** — 29 von 102 Zellen. Ziel dieses Plans:
Aufteilung in zwei Kapitel und rund 30 % weniger Text, ohne einen einzigen Befund zu
verlieren.

---

## Teil A — Kapitel 7 in zwei Kapitel teilen

Die Bruchlinie liegt zwischen zwei Fragen, die heute vermischt sind:

- **„Was ist die Regel, und funktioniert sie?"** — §7.1, §7.4
- **„Unter welchen Bedingungen trägt sie?"** — §7.5, §7.6, §7.7, §7.8

Dazwischen stehen §7.2 und §7.3, die weder das eine noch das andere sind: sie begründen
*rückblickend*, warum die Regel so aussieht, wie sie aussieht.

### Vorschlag

**Kapitel 7 — Der sequenzielle Workflow** *(„so geht es")*

| neu | heute | Inhalt |
|---|---|---|
| 7.1 | 7.1 | Die Entscheidungsregel (Werkzeugblock → Bibliothek, Teil D) |
| 7.2 | 7.4 | Der Workflow auf den echten Modellen |
| 7.3 | 7.4-Unterabschnitt | Verlauf der Kenngrößen mit k |

**Kapitel 8 — Wo die Regel trägt** *(„und wann nicht")*

| neu | heute | Inhalt |
|---|---|---|
| 8.1 | 7.2 + 7.3, stark gekürzt | Warum nicht einfacher? Zweiseitig und ohne Checkpoints |
| 8.2 | 7.5 | Der Abstand ρ und die Wahl der Schranke |
| 8.3 | 7.6 | Die Unsicherheit der Schranke |
| 8.4 | 7.8 | Der Arbeitspunkt q |
| 8.5 | 7.7 | Regime-Übersicht und Konsequenz |

**Konsequenz für die Nummerierung:** die heutigen (leeren) Kapitel 8–11 werden zu 9–12.
Anker, Inhaltsverzeichnis und §1.6 müssen nachgezogen werden — dasselbe Skript wie beim
letzten Umbau.

**Warum diese Reihenfolge in Kapitel 8:** §8.1 motiviert, §8.2–§8.4 kalibrieren jeweils
einen Bestandteil (Schranke, Bandbreite, Arbeitspunkt), §8.5 fasst zusammen. Heute steht
die Konsequenz (§7.7) *vor* dem Arbeitspunkt (§7.8) und muss deshalb vorwärts verweisen —
das löst sich damit auf.

---

## Teil B — Textkürzungen, zellenweise

Die zehn größten Markdown-Zellen und was mit ihnen passieren soll.

| Zelle | Abschnitt | jetzt | Ziel | Was gestrichen wird |
|---|---|---|---|---|
| 94 | §7.7 Konsequenz | **943** | ~250 | Wiederholt sämtliche Zahlen aus den Tabellen darüber. Behalten: Regime-Tabelle, vier Stichpunkte, offene Punkte. Streichen: die ausformulierten Prozentwerte, die Budget-Tabelle (steht in §7.4), die Wiederholung der Fehlerarten-Definition. |
| 88 | §7.5 Lesart | **720** | ~330 | Der Absatz „Warum überhaupt die Schranke verglichen…" steht inhaltsgleich im Docstring von `cmax_skew_vec`. Ersatzlos streichen. Ebenso die Wiederholung der Sweep-Zahlen. |
| 97 | §7.8 Ergebnis | 568 | ~300 | Tabelle behalten, die drei ausformulierten Befunde auf je zwei Sätze. |
| 21 | §2.3 CGF | 438 | ~280 | Der Kasten „Wessen Verteilung ist das?" und die Aufzählung „Zwei Eigenschaften" sind pädagogisch, aber die Herleitung trägt ohne sie. |
| 60 | §6.1 Standardfehler | 410 | ~330 | Die vier Schritte sind eine echte Herleitung → behalten. Kürzen: die anschaulichen Zwischenkommentare („Anschaulich: die Varianz ist aus quadrierten Abweichungen gebaut…"). |
| 93 | §7.6 Ergebnis | 353 | ~200 | Die zwei „Lesehinweis"-Kästen zusammenziehen. |
| 86 | §7.5 Einleitung | 343 | ~200 | ρ-Definition und Modell-Tabelle behalten, die Motivation auf drei Sätze. |
| 20 | §2.2 Parameter c | 332 | ~200 | Der Kasten „Warum die $v_i$ den Index behalten" wiederholt die Zeile darüber in Worten. Streichen. |
| 78 | §7.3 Lesart | 305 | → §8.1, ~120 | Siehe Teil C. |
| 34 | §3.2 A2 | 280 | ~180 | Vier Absätze mit demselben Kern („der ungesehene Rand"). Auf zwei zusammenziehen. |

**Weitere Kandidaten unter 250 Wörtern**, gleiche Behandlung: Zelle 28 (brentq-Erklärung,
235 → 40, siehe Teil C), Zelle 57 (§5.4, 199 → 120), Zelle 64 (§6.2, 182 → 120),
Zelle 42 (§4.2 Paradox-Tabelle, 215 → 150, Tabelle behalten, Fließtext halbieren).

---

## Teil C — Ganze Passagen streichen

**1. §7.3 als eigener Abschnitt entfällt** *(heute 215 + 499 Code + 305 = 1019 Wörter)*

Der synthetische Grenzfall mit Bonferroni-Checkpoints ist ein historischer Zwischenstand.
Sein einziger bleibender Befund: *die naive Regel misfeuert bei ρ ≈ 1 in 47,6 % der
Sequenzen, Bonferroni drückt das auf 11,5 %, lässt aber 39,5 % unentschieden.* Das sind
zwei Zeilen einer Tabelle.

→ Zusammen mit §7.2 zu **§8.1 „Warum nicht einfacher?"**: eine Code-Zelle, eine Tabelle,
~150 Wörter. **Ersparnis: ~1200 Wörter.**

**2. Die brentq-Erklärung (Zelle 28, 235 Wörter)**

Ein Tutorial über Bisektion, Sekantenverfahren und inverse quadratische Interpolation.
Das ist Numerik-Lehrbuchstoff, kein Projektinhalt — und seit dem Umbau benutzt
`cmax_skew` ohnehin ein Gitter zur Einklammerung plus brentq. → Auf zwei Sätze
(*warum* eingeklammert wird, nicht *wie* Brent funktioniert).

**3. Die Fehlerarchäologie in Markdown → in `AENDERUNGEN_kap7_kap8.md`**

Über das Kapitel verteilt stehen mehrere Passagen der Form „eine frühere Fassung hatte X
falsch gemacht":

- §7.3, Kasten zum Zirkelschluss bei der Wahrheitsdefinition (~110 Wörter)
- §7.5, „eine Zwischenfassung dieses Notebooks hatte es andersherum…" (~60)
- §7.6, „Rückwirkung auf §6.2" (~70)
- §7.1-Wegweiser, „Eine frühere Fassung hatte γ₂ aus der vollen Stichprobe…" (~40)

Die sind für den Bericht wertvoll, im Notebook aber Ballast — und die Datei
`AENDERUNGEN_kap7_kap8.md` existiert genau dafür. → Dort sammeln, im Notebook je ein
Halbsatz mit Verweis. **Ersparnis: ~280 Wörter**, und die Korrekturen stehen an *einer*
Stelle statt verstreut.

**4. Der Wegweiser (Zelle 70, 141 Wörter)**

Nach der Aufteilung in zwei Kapitel mit je eigener kurzer Einleitung ist er redundant.
→ Streichen.

**5. Kapitel 3 (554 Wörter, null Code)**

Das einzige Kapitel ohne eine einzige Rechnung. Inhalt: zwei Annahmen. → Tabelle mit
zwei Zeilen plus je einem Absatz, ~250 Wörter. **Ersparnis: ~300.**

---

## Teil D — Werkzeugblock in die Bibliothek

Zelle 72 ist mit **938 Wörtern Code** die größte Zelle des Notebooks und enthält keine
Argumentation, sondern Infrastruktur: `cmax_skew`, `cmax_skew_vec`, `log_neff_ratio`,
`run_stats`, `run_stats_2d`, `se_c`, `decide_naive`, `fail_only_2d`, `stat_D`,
`monitor_boot`.

→ Neues Modul **`src/uq_mace/screening.py`**. Das Notebook behält:

```python
from uq_mace.screening import (
    CHECKPOINTS, K_FLOOR, Q_ALPHA, cmax_skew, cmax_skew_vec,
    log_neff_ratio, monitor_boot, run_stats, se_c, stat_D,
)
```

plus die beiden Verifikationsausgaben (brentq-Vergleich, `run_stats_2d` gegen
`run_stats`) — die gehören sichtbar ins Notebook, weil sie Befunde sind.

Drei Gewinne: die Erzählung wird um ~800 Wörter Code leichter, die Funktionen werden in
`tests/` testbar, und `analyses/13_sequential_screening/operating_characteristic.py`
kann importieren statt zu duplizieren (dort liegen dieselben Funktionen aktuell ein
zweites Mal).

---

## Teil E — Redaktionsregeln, damit es schlank bleibt

Die eigentliche Ursache der Textmasse ist ein Muster, nicht einzelne Zellen. Vier Regeln:

1. **Höchstens ein Absatz Lesart je Code-Zelle**, Richtwert 120 Wörter. Heute folgen auf
   manche Zelle drei Absätze plus zwei Kästen.
2. **Keine Zahl wiederholen, die in der Ausgabe direkt darüber steht.** Der Text
   interpretiert, er referiert nicht. Das allein streicht geschätzt 400 Wörter in
   Kapitel 7.
3. **Begründungen für Code-Entwurf gehören in den Docstring**, nicht ins Markdown. Der
   Fall `cmax_skew_vec` zeigt es: die Begründung steht bereits zweimal da.
4. **Historische Irrwege: ein Satz mit Verweis.** Der Weg gehört in die Änderungsdoku,
   nicht in die Darstellung des Ergebnisses.

---

## Erwartetes Ergebnis

| | jetzt | nach Umbau |
|---|---|---|
| Wörter gesamt | 17 743 | ~12 300 |
| davon Kapitel 7 (+8) | 9 045 | ~5 000 |
| Zellen | 102 | ~92 |
| Anteil Prosa | 66 % | ~58 % |

Kein Befund und keine Abbildung geht verloren; gestrichen werden Wiederholungen,
Doppelbegründungen, Numerik-Tutorials und die Fehlerarchäologie.

---

## Reihenfolge der Umsetzung

1. **Teil D** zuerst (Werkzeugblock → Bibliothek). Rein mechanisch, ändert keine Zahl,
   verkleinert aber die Zelle, die alle folgenden Schritte im Weg steht.
2. **Teil A** (Aufteilung + Umnummerierung 8–11 → 9–12), inklusive Anker,
   Inhaltsverzeichnis und §1.6.
3. **Teil C** (Streichungen) — dabei die Fehlerarchäologie nach
   `AENDERUNGEN_kap7_kap8.md` umziehen.
4. **Teil B** (Kürzungen zellenweise), zuletzt, weil sich durch 1–3 einige Zellen ohnehin
   erübrigen.
5. Restart & Run All, danach die Zahlen in den gekürzten Texten gegen die neuen Ausgaben
   prüfen.

**Offene Entscheidung für dich:** ob §8.1 („Warum nicht einfacher?") überhaupt bleiben
soll. Argument dafür: ohne sie steht die Regel unbegründet im Raum, und die 47,6 % sind
der einzige harte Beleg für die Mehrfachtest-Inflation. Argument dagegen: für einen Leser,
der nur die Methode anwenden will, ist es Ballast. Meine Empfehlung: behalten, aber auf
eine Zelle plus Tabelle eindampfen.
