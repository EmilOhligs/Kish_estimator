# Überarbeitung Kapitel 7 & 8 — was geändert wurde und warum

> **Nachtrag: Umbau der Struktur.** Nach der ersten Überarbeitungsrunde wurde das alte
> Kapitel 7 (Entscheidungsschranke) **komplett gestrichen** und Kapitel 8 zum neuen
> Kapitel 7. Die Herleitung beider Schranken steht ohnehin schon in §2.6; die
> Falsch-FAIL-Zone wird jetzt dort behandelt, wo sie zählt — in §7.5, gemessen statt
> definiert. Kapitel 9–12 sind zu 8–11 geworden, Inhaltsverzeichnis und §1.6 nachgezogen.
>
> Weiter gestrichen: der Vergleich mit/ohne `k ≥ 5` (alt §8.4) und der Einzelblick
> (alt §8.5). Neu ist §7.4: der **Workflow-Test mit Checkpoints** auf den echten Modellen,
> mit c, c_max, ρ, exaktem N_eff und Urteil je Lauf.
>
> Alle Definitionen liegen jetzt in **einem Werkzeugblock** (§7.1) statt über sechs Zellen
> verstreut. Notebook: 114 → 99 Zellen, Laufzeit 76 s, 0 Fehler, 0 unaufgelöste Namen.
>
> Neue Gliederung: 7.1 Entscheidungsregel · 7.2 naive zweiseitige Regel · 7.3 synthetischer
> Grenzfall · **7.4 Workflow auf den echten Modellen** · 7.5 Abstand und Schranke (ρ) ·
> 7.6 Unsicherheit der Schranke · 7.7 Konsequenz.

Backup des Vorzustands: `screening_methode_BACKUP_vor_ueberarbeitung.ipynb`
Stand: Notebook läuft Restart + Run-All in ~35 s durch, 0 Fehler, 0 unaufgelöste Namen,
27/27 Code-Zellen mit Output, 18 eingebettete Abbildungen.

---

## 1. Lauffähigkeit (war blockierend)

Drei Namen wurden vor ihrer Definition benutzt — das Notebook lief nur in einer Session mit
Altbestand im Namensraum.

| Zelle | fehlte | behoben durch |
|---|---|---|
| §8.2 | `cmax` | `R`, `LNR`, `cmax` stehen jetzt im Werkzeugblock am Anfang von §8 |
| §8.3 | `norm` | `from scipy.stats import norm` ergänzt |
| §8.5 | `rng2` | eigener Strom `np.random.default_rng(5)` in der Zelle definiert |

Zusätzlich stand `fail_only` in §8.6, wurde aber schon in §8.5 gebraucht → in den
Werkzeugblock verschoben. Duplizierte Definitionen von `running_c`/`decide_naive`
(§8.2 *und* §8.4) entfernt.

**Prüfung:** AST-basierte Def-vor-Use-Analyse über alle Code-Zellen, 0 Befunde.

---

## 2. Der wichtigste inhaltliche Fehler: Look-ahead im Monitor

§8.2/§8.4/§8.6 haben `γ₂` aus der **vollen** Stichprobe berechnet und ab k=5 in die
SE-Formel gefüttert — der Monitor benutzte also Zukunftsinformation genau in der Größe, die
seine Bandbreite bestimmt. Das ist jetzt durchgängig kausal: jede Größe bei Punkt k, auch
γ₁, γ₂ und die daraus gebildete Schranke, kommt aus `d[:k]` (über `running_moments`).

Nebenbei behoben: `dE.std()` (ddof=0) in §8 gegen `ddof=1` im Rest des Notebooks.

---

## 3. Die Schranke — Umstellung auf `c_max^schief`, kausal geschätzt

Entscheidung aus der Rückfrage. §8.2, §8.3, §8.4, §8.5 und §8.6 entscheiden jetzt gegen die
schiefe-korrigierte Schranke je Modell statt gegen `c_max^Gauß`.

**Warum das mehr ist als Kosmetik** (neue Zellen in §8.6, Teil 2): gemessen gegen eine
*schrankenfreie* Wahrheit (exaktes Kish, N = 400 000) verwirft die Gauß-Schranke ein Modell,
das die Effizienzbedingung nachweislich erfüllt (N_eff/n = 0,832), in **33 %** der
Sequenzen. Die Falsch-FAIL-Zone aus §7.2 ist unter der einseitigen Regel keine Randnotiz,
sondern die dominierende Fehlerquelle.

| c_true/c_max | N_eff/n | wahr | Gauß fix | schief Orakel | schief kausal |
|---|---|---|---|---|---|
| 0,70 | 0,891 | PASS | 2,1 % | 0,7 % | 1,6 % |
| 0,80 | 0,863 | PASS | 7,2 % | 2,3 % | 4,9 % |
| 0,90 | 0,832 | PASS | **33,4 %** | 7,1 % | 14,9 % |
| 1,00 | 0,801 | PASS | 95,1 % | 29,1 % | 41,4 % |
| 1,10 | 0,768 | FAIL | 100 % | 90,5 % | 91,7 % |

**Zwei Fallstricke, die dabei auftraten und im Notebook dokumentiert sind:**

1. `cmax_skew` mit `brentq` auf `[1e-4, 3]` **schlägt für γ₂ < 0 fehl** — die Quartik kippt
   bei großem c wieder unter null, das Bracket hat keinen Vorzeichenwechsel, obwohl die
   gesuchte kleinste Wurzel existiert. Trat real auf (mace-L0-01, γ₂ = −0,23). `cmax_skew`
   bracket jetzt bis zum Maximum von f und fällt sonst dokumentiert auf Gauß zurück.
2. Der naheliegende Kurzschluss — das Kriterium direkt bei `c_lo` auswerten statt die
   Schranke zu bestimmen — ist **nur nahe der Schranke** äquivalent. Bei c ≈ 1,2 (die realen
   L0-Modelle) ist die Kumulantenreihe wertlos und behauptet PASS. In einer Zwischenfassung
   führte das zu **51 % Fehlurteilen**. Jetzt: Schranke bei c ≈ 0,5 auswerten (konvergent,
   §2.6), c_lo damit vergleichen. Neu: `cmax_skew_vec` (Newton ab der Gauß-Lösung,
   vektorisiert, gegen `brentq` auf 2·10⁻¹³ verifiziert).

---

## 4. §8.6 — die Begründung war falsch herum

**Alt:** „Bei durchweg gemessener Rechtsschiefe (γ₁ > 0) ist eine Fluktuation nach oben die
unwahrscheinlichere Richtung" → deshalb keine Mehrfachtest-Korrektur nötig.

**Neu, per Spiegeltest** (ΔE → 2μ − ΔE kippt γ₁, lässt c und γ₂ unangetastet):

| | Gauß fix | schief Orakel | schief kausal |
|---|---|---|---|
| γ₁ = +0,501 | 33,7 % | 7,65 % | 15,1 % |
| γ₁ = −0,501 | 2,7 % | 7,65 % | 4,6 % |

Rechtsschiefe ist die **gefährliche**, nicht die schützende Seite. Der alte Schluss
verwechselte die Schiefe von ΔE mit der Stichprobenverteilung von c(k). Was γ₁ tatsächlich
tut, ist die **Schranke verschieben** — die Orakel-Zeile ist exakt symmetrisch (7,65 % gegen
7,65 %), die Vorzeichenabhängigkeit ist vollständig ein Artefakt der falschen Schranke.

**Damit ist die offene Frage aus dem Briefing beantwortet:** bei γ₁ < 0 kippt nichts,
sofern man die Schranke mitschätzt.

**Und die Korrekturfreiheit gilt nur mit Abstand:** 1,6 % bei 30 % Abstand, 4,9 % bei 20 %,
14,9 % bei 10 %, 41,4 % direkt auf der Schranke. Die pauschale Aussage „braucht keine
Mehrfachtest-Korrektur" ist gestrichen.

---

## 4b. Neuer Abschnitt §8.7 „Die Unsicherheit der Schranke" (+ Umnummerierung 8.7→8.8)

Bis dahin deckte das z-Band nur SE(c) ab — verglichen wird aber mit ĉ_max, und die wird aus
denselben k Punkten geschätzt. Neu dokumentiert und gerechnet:

- **c_max hängt nur von (γ₁, γ₂, R) ab, nicht von c.** Daher kommt die gesamte
  k-Abhängigkeit über SE(γ̂₁), SE(γ̂₂). Sensitivitäten: ∂c_max/∂γ₁ = +0,191,
  ∂c_max/∂γ₂ = −0,059 — γ₁ trägt rund doppelt so stark.
- **SE(ĉ_max) ist ab k ≈ 10 so groß wie SE(ĉ)** — die Schranke ist eine gleichwertige
  zweite Rauschquelle, keine feste Referenz.
- **Korr(ĉ, ĉ_max) = 0,04…0,26**, schwach positiv; die Quadratur überschätzt SE(D) um
  1–13 %, ist also konservativ.
- **Gegen die analytische Fortpflanzung:** √(6/k) sagt bei k=5 eine Standardabweichung von
  1,10 für γ̂₁ voraus, während |γ̂₁| ≤ (k−2)/√(k−1) = 1,5 beschränkt ist und real 0,61
  streut. Deshalb wird die Entscheidungsgröße D = ĉ − ĉ_max **direkt gebootstrappt**.
- **Neue Regel:** FAIL wenn D(k) > z·SE_boot(D(k)), ausgewertet an 12 geometrischen
  Checkpoints statt bei jedem k.

Wirkung (1000 Sequenzen, identische Ziehungen für beide Regeln):

| ρ | wahr | §8.6 | §8.7 (Bootstrap) |
|---|---|---|---|
| 0,80 | PASS | 4,8 % | **0,9 %** |
| 0,90 | PASS | 15,4 % | **3,6 %** |
| 0,95 | PASS | 23,8 % | **7,6 %** |
| 1,10 | FAIL | 91,5 % | 80,4 % |
| 1,20 | FAIL | 100 % | 99,1 % |

Grauzone schrumpft von ρ ∈ [0,8; 1,2] auf [0,9; 1,2]. Reale Modelle unberührt (L0 200/200
bei k_med = 5, L2 null Fehlalarme). Notebook-Laufzeit dadurch 35 s → 80 s.

**Nicht getrennt gemessen** (im Notebook als Vorbehalt vermerkt): wie viel vom Gewinn auf
das Bootstrap-Band und wie viel auf die Reduktion von ~400 auf 12 Blicke entfällt.

## 4c. ρ als Symbol eingeführt

ρ = c/c_max war bisher nur implizit als Tabellenspalte vorhanden. Jetzt in §8.6 Teil 2
definiert, mit der Klarstellung, dass ρ **nicht** auf [0,1] beschränkt ist — ρ ≤ 1 ist das
Kriterium (⟺ N_eff/n ≥ R), keine Nebenbedingung. Dazu eine Tabelle der vier realen Modelle:
ρ = 0,62 / 0,64 (L2) und 2,19 / 2,47 (L0).

§8.8 beginnt jetzt mit einer expliziten **Regime-Übersicht** (sicher PASS ρ ≲ 0,9 |
Grauzone 0,9–1,2 | sicher FAIL ρ ≳ 1,2) und der Formel, warum der Fehler gerade bei ρ ≈ 1
sitzt: der Sicherheitsabstand (c_max − c_true) verschwindet, übrig bleibt z·SE pro Blick.

---

## 5. §7 — die γ₁-Vorzeichen-Behauptung

§7.4 sagte, die Gauß-Schranke sei konservativ „solange γ₁ ≥ 0". Die eigene Tabelle in §7.3
zeigte aber schon, dass die Zonenbreite bei γ₁ = 0 bereits negativ ist. Der Nulldurchgang
liegt bei γ₁* ≈ (7/12)·γ₂·c_max — numerisch **+0,1075**, analytische Näherung **+0,1075**
(exakte Übereinstimmung, jetzt im Notebook geprüft und im Plot markiert). Text in §7.3/§7.4
entsprechend korrigiert; mace-L0-c-01 liegt mit γ₁ = +0,207 nicht mit viel Abstand darüber.

---

## 6. §8.5 — als widerlegte Vereinfachung markiert

Entscheidung aus der Rückfrage. Neuer Teil 3 in der Zelle: eine Sequenz, die 50 Punkte lang
gutartig ist (c = 0,449) und dann einbricht (c = 0,846) — genau das A2-Szenario aus §3.2.

- Einzelblick bei k = 30: erkannt in **0,2 %** der Sequenzen
- laufende Überwachung: erkannt in **100 %**

§8.7 empfiehlt jetzt eindeutig §8.6.

---

## 7. Weitere korrigierte Stellen

- **§8.2 „98 % DFT gespart"** galt auch für die PASS-Modelle — unter der FAIL-only-Regel
  sparen die per Konstruktion **null**. Spalte differenziert; zusätzlich die Zahl, auf die
  es ankommt: gegen die 5000 Produktionspunkte sind es **99,9 %**, und die Budget-Trennung
  400/500 vs. 5000 steht jetzt explizit in §8.7.
- **§8.4 ohne k ≥ 5:** war „4 von 30 falsch bei k = 2", ist jetzt **20 von 30 bei k = 1** —
  bei einem Punkt ist c gar nicht definiert und die Regel behauptet PASS, bevor sie etwas
  gesehen hat. Deutlicheres Argument für die Untergrenze.
- **„in allen fünf Modellen, §7.1"** → vier (das fünfte steht nur im Briefing).
- **Hängender Verweis** „§6.4: γ₁ stabil ab k ≈ 210" (steht dort nicht) → auf §6.1 und die
  SE-Formel umgestellt.
- **k̂ > 0,5 in §8.4** wird nicht mehr weggewinkt: der Kish-Wert existiert dann als Zahl,
  aber nicht als Schätzer.
- **§8.3-Wahrheit:** N = 300 000 beseitigt nur MC-Rauschen; gezogen wird weiter aus 400
  Werten. Diese Einschränkung steht jetzt dabei.
- **`\tfrac`** in einem matplotlib-Titel (mathtext kennt es nicht) → `\frac`.
- **k̂-Verläufe** unter k = 50 blass gezeichnet (zu wenig Tail für den Schätzer).

---

## 8. Offen — bewusst nicht angefasst

1. **Autokorrelation.** Alles ist iid gebootstrappt; ein realer MD-Strom ist korreliert,
   dann ist SE(c) um √τ größer und der Monitor zu selbstsicher. Bei k_med = 5 potenziell
   entscheidend. In §8.7 jetzt als Vorbehalt benannt (alle k als *unabhängige* Punkte lesen,
   im MD-Betrieb mit dem Thinning-Intervall multiplizieren) — ein AR(1)-Sweep gehört in §11.
2. **Kapitel 1–6** wurden nicht überarbeitet. Zwei Stellen sind durch die §8-Änderung
   faktisch überholt:
   - §1.6 spricht von „den drei Annahmen A1–A3", es sind zwei.
   - §6.2 sagt, der Kleine-k-Bias von γ₁,γ₂ zeige „in die sichere Richtung, nie fälschlich
     PASS". Unter der einseitigen Regel ist genau das die einzige Fehlerart — die Richtung
     hat sich gedreht. §8.6 enthält dazu einen Rückverweis, §6.2 selbst ist unverändert.
3. **§4.2, Zeile L2-c-01 / n = 400** — die Zahlen (W = 0,988, p = 3·10⁻³, −1,4 %) werden von
   keiner Zelle erzeugt; `SHOW_400` enthält das Modell nicht.
4. **Reales Modell in der Falsch-FAIL-Zone** fehlt weiterhin; die Zone ist nur synthetisch
   demonstriert (c ≈ 0,47–0,53 wäre der Testfall).
5. **§9–§12** sind unverändert leere Platzhalter, auf die §3.2, §5.3 und §8.4 verweisen.
