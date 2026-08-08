# Review Runde 4 — Text und Code getrennt

Stand: 91 Zellen, 95 s, 0 Fehler, `check_notebook.py` **alle sieben Invarianten grün**.

Die Befunde dieser Runde sind durchweg solche, die der Checker strukturell nicht sehen
kann: Markdown-Rendering, inhaltliche Widersprüche, Linearisierungen. **Fünf der neun
Textbefunde habe ich in der letzten Runde selbst eingebaut** — vier davon durch unvollständige
Korrekturen.

---

# Teil 1 — Text

## Von mir in Runde 3 eingeschleppt

### N1. Titel: verschachteltes Bold kehrt die Betonung um **[eingeschleppt]**

```
**Kann man ... erkennen, dass das Reweighting eines L0-Modells **nicht** tragen wird ...?**
```

Vier `**`-Marker in Folge ergeben zwei Bold-Spans mit einer Lücke dazwischen. Gerendert
wird also **alles fett außer dem Wort „nicht"** — genau das Wort, das betont werden sollte.
Fix: inneres Hervorheben zu `*nicht*` oder das äußere Bold weglassen.

### N2. §6.3 und §6.4 widersprechen dem gerade korrigierten §6.2 **[eingeschleppt]**

In Runde 3 habe ich §6.2 korrigiert („unter der einseitigen Regel zeigt der Bias ins
Unsichere"). Drei Stellen sind stehengeblieben:

- §6.3-Lesart: „und der Bias zeigt in die sichere Richtung."
- §6.4, Stichpunkt 2: „noch Entscheidung (Bias in die sichere Richtung)."
- §6.4, Schlusssatz: „die Formkorrekturen verschieben nur die Schranke — kontrolliert und
  **konservativ**."

§6.4 sagt damit zwei Zellen nach der Korrektur wieder das Gegenteil. Dieselbe Fehlerart wie
T3 in Runde 3, nur jetzt selbstverschuldet: **eine Aussage korrigiert, ihre Wiederholungen
nicht gesucht.**

### N3. §3.2: doppelte Leerzeile und eine überlange Zeile **[eingeschleppt]**

Beim Wiedereinsetzen der Gegenbeispiele blieb `\n\n\n` stehen, und der Satz „Übrig bleibt
$E[w^2]=…$. Diese Bedingung bindet nur die linke Flanke…" wurde zu einer 110-Zeichen-Zeile
zusammengezogen. Kosmetisch, aber der Rest des Notebooks hält 92 Zeichen.

### N4. §8.3: historische Aussage ohne Beleg **[eingeschleppt]**

> „Bei einem frühen ersten Blick trennte dieser fehlende Term Orakel- und kausale Variante
> deutlich; ab $k=n/10$ liegen beide gleichauf (§8.2)"

Die erste Hälfte ist im Notebook **nirgends** gemessen — §8.2 zeigt nur den zweiten Zustand.
Ich habe damit meinen eigenen Befund aus Runde 1 wiederholt (unbelegte Zahl), nur ohne Zahl.
Entweder als historisch kennzeichnen und nach `AENDERUNGEN` verschieben, oder streichen.

## Ältere Befunde, noch offen

### N5. §5.2: „$\hat k<0{,}5$ **garantiert** die Existenz" **[seit Runde 1]**

Widerspricht §3.2 direkt: „Momentexistenz ist eine Aussage über den ungesehenen Rand […]
beweist nichts." $\hat k$ ist ein Schätzer mit SE ∝ $n^{-1/4}$. Ersetzen durch „ist die
einzige verfügbare Evidenz für".

### N6. §1.3 widerspricht §5.2 **[neu]**

§1.3: „Ihre Verteilung fasst **ein einziger dimensionsloser Parameter** zusammen."
§5.2: „Bei festem $c$ ist $N_\text{eff}/n$ **nicht** bestimmt."

Beides kann nicht stimmen. §1.3 sollte „**skaliert**" sagen statt „fasst zusammen" — $c$ ist
die Skala, die Form kommt dazu.

### N7. Titel: „klares PASS" und „exakte Kumulantenformel" **[neu]**

L2c wird als „gutartiger Kontrast (klares PASS, exakte Kumulantenformel)" beschrieben. Die
Regel behauptet nie PASS (§7.1), und die Kumulantenformel ist eine abgebrochene Reihe — für
L2c *sehr genau* (< 0,1 %), aber nicht exakt.

### N8. §5.2: `c_max` unformatiert **[neu]**

„Hieraus wird die Schwelle c_max berechnet." — im Rest des Notebooks durchgehend
$c_\text{max}$.

### N9. §6.2 nennt SE-Formeln, die dieselbe Zelle verwirft **[neu]**

„$\gamma_1$ ($\mathrm{SE}\approx\sqrt{6/k}$) und $\gamma_2$ ($\mathrm{SE}\approx\sqrt{24/k}$)
folgen derselben $\sqrt{1/k}$-Rate" — richtig ist die **Rate**, falsch sind die Vorfaktoren
(nur unter Normalität gültig, was die Zelle drei Zeilen später numerisch widerlegt). Ein
Halbsatz genügt: „…derselben Rate; die Vorfaktoren gelten allerdings nur unter Normalität."

---

# Teil 2 — Code

## Look-ahead: erneut kein Befund

Alle 23 Zellen nochmals durchgegangen, mit Schwerpunkt auf den sechs in Runde 3 geänderten
(24, 35, 38, 60, 70, 86). Der Entscheidungspfad bleibt kausal:

- **Zelle 24** und **60** sind Verifikations- bzw. Diagnosezellen ohne Entscheidung.
- **Zelle 70**: `cmax_reps` sammelt jetzt je Lauf die Schranke aus `g1[-1], g2[-1]` — das ist
  eine **Plot-Annotation** auf Modellebene, keine Entscheidungsgröße. Der Median über zehn
  Läufe ist korrekt umgesetzt.
- **Zelle 38/86**: reine Darstellung.

Ebenfalls geprüft: Funktionsparameter, die globale Namen tragen (`R` in `cmax_exact`, `c`,
`k`, `beta`, `q` in weiteren). Alle lokal gebunden — **keine Kollision**, anders als bei der
Schleifenvariable aus Runde 3.

## Befunde

### C1. Zelle 24: `c_hi=0.7155` zweifach hartkodiert **[eingeschleppt]**

Das ist `C_VALID` aus `uq_mace.screening`, hier als nackte Zahl — zweimal, in `cmax_exact`
und `_cmax_of`. Ändert sich `R5_TOL` in der Bibliothek, driften Notebook und Modul
auseinander, ohne dass etwas bricht. Die Zelle kann `C_VALID` nicht importieren (der
Werkzeugblock kommt erst in §7.1), sollte den Wert aber aus einer lokalen Konstante
ableiten: `R5_TOL = 0.05; C_VALID = 0.5*(120*R5_TOL)**0.2`.

### C2. Zelle 24: `cmax_exact` sieht allgemein aus, ist es aber nicht **[eingeschleppt]**

`cmax_exact(R, c_hi)` und `_cmax_of(g1v, g2v=g2, …)` schließen über die modulweiten `g1`,
`g2` von `ensemble_L2c`. Für die Zelle korrekt, aber die Signatur suggeriert Allgemeinheit.
`g1`, `g2` sollten Parameter sein.

### C3. Zelle 60: die Prozentangabe ist linearisiert **[alt, nie erwähnt]**

`se3 = c**3 * SEg1` wird als „% auf $N_\text{eff}$" gedruckt. Tatsächlich ist der Effekt
$e^{se_3}-1$. Bei der größten Zeile: gedruckt 13 %, exakt 14,3 %. Systematischer Versatz von
gut einem Prozentpunkt, nirgends vermerkt. Entweder exakt rechnen oder die Näherung
dazuschreiben.

### C4. Zelle 60: Ternär-Print mit zwei fast gleichen Zweigen **[alt]**

Die beiden Zweige unterscheiden sich nur in `.1f` gegen `.0f`. Altlast aus der
ursprünglichen Fassung.

---

# Bewertung

Der Checker hat in Runde 3 gehalten, was er sollte — die Fehlerklassen, für die er gebaut
wurde (Verweise, Anker, Zahlen ohne Beleg, Nummerierung, Schleifenvariablen), sind sämtlich
grün und in dieser Runde nicht wieder aufgetreten.

Die verbleibende Lücke ist eine andere: **wenn ich eine Aussage korrigiere, suche ich ihre
Wiederholungen nicht.** N2 ist genau das, und es ist dieselbe Mechanik wie T1/T3 in Runde 3.
Ein Checker kann das nicht fangen — aber die Suche nach Schlüsselphrasen vor dem Abschluss
einer Korrektur schon. Für N2 hätte `sichere Richtung|konservativ` gereicht; ich hatte diese
Suche in Phase 2.4 des Plans sogar vorgesehen und dann nicht ausgeführt.

## Priorität

1. **N1** — Titel rendert die Kernaussage verkehrt herum.
2. **N2** — drei Stellen, direkter Widerspruch innerhalb von Kapitel 6.
3. **N5, N6** — inhaltliche Widersprüche zwischen Kapiteln.
4. **C1, C2** — Bibliothek und Notebook können auseinanderdriften.
5. **N4, N7, N9, C3** — überzogene oder unbelegte Formulierungen.
6. **N3, N8, C4** — Kosmetik.
