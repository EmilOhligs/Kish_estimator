# Was der Code tun soll — Referenz zum Gegenlesen

Übersicht für die Code-Durchsicht. Zu jeder Funktion steht hier, **was sie
leisten soll**, nach welcher Formel, und woran eine Abweichung auffiele. Gedacht
zum Danebenlegen, während ein anderes Modell den Code Zeile für Zeile erklärt:
passt die Erklärung zu dem, was hier steht?

Die Contracts sind meine Lesart aus Code und Notebook-Text. Wo Erklärung und
Contract auseinandergehen, ist **nicht automatisch die Erklärung falsch** — es
kann auch sein, dass ich den Zweck falsch verstanden habe oder der Code etwas
anderes tut als beabsichtigt. Stellen, bei denen ich selbst unsicher bin, sind
mit **?** markiert.

---

## 1. Worum es geht

Reweighting korrigiert MACE-Mittelwerte auf DFT-Niveau mit Gewichten
$w_i = e^{-\beta\Delta E_i}$, wobei $\Delta E = E_\text{DFT}-E_\text{MACE}$. Wie
gut das statistisch trägt, misst der Kish-Wert

$$N_\text{eff} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}, \qquad
\text{Ziel: } N_\text{eff}/n \ge R .$$

Zentrale Größe ist $c=\beta\,\mathrm{std}(\Delta E)$. Über die
Kumulantenentwicklung

$$\log(N_\text{eff}/n) = -c^2 + \gamma_1c^3 - \tfrac{7}{12}\gamma_2c^4 + O(c^5)$$

wird aus dem Ziel eine Schranke: $N_\text{eff}/n\ge R$ gilt genau dann, wenn
$c\le c_\text{max}$, und $c_\text{max}$ ist die kleinste positive Wurzel der
Quartik $c^2-\gamma_1c^3+\tfrac{7}{12}\gamma_2c^4 = -\ln R$.

**Die Entscheidungsregel des Workflows:** FAIL, sobald

$$\hat c(k) - \mathrm{SE}(\hat c) \;>\; \hat c_\text{max}(k) + \mathrm{SE}(\hat c_\text{max}),$$

geprüft an Checkpoints ab $k=n/10$. Einseitig — PASS wird nie behauptet, weil ein
frühes PASS keine Rechenzeit spart. Damit gibt es genau eine Fehlerart: ein
brauchbares Modell abbrechen.

### Aufrufkette — sinnvolle Lesereihenfolge

```
reweighting_weights ──> effective_sample_size          die Grundgrößen
                   └──> psis_khat                       das Gate

running_moments ──> run_stats / run_stats_2d           laufende Momente
                                    │
cmax_gauss ──> cmax_skew ──> cmax_skew_vec ──┐         die Schranke
                                              ├──> stat_D
se_c ─────────────────────────────────────────┴──> monitor_split   die Regel
```

---

## 2. `src/uq_mace/screening.py`

### Konstanten

| Name | Wert | soll bedeuten |
|---|---|---|
| `K_FLOOR` | 50 | erster Blick bei 10 % der geplanten Punkte |
| `R5_TOL` | 0.05 | zulässiges Restglied der Kumulantenreihe |
| `C_VALID` | $\tfrac12(120\cdot$`R5_TOL`$)^{1/5}\approx0{,}716$ | darüber ist die Reihe wertlos |
| `Q_ALPHA` | 1.64 | einseitiges 5-%-Normalquantil, **nur noch** für die alten Vergleichsfassungen |

`C_VALID` kommt aus $(2c)^5/5!\le$ `R5_TOL`. Der effektive Entwicklungsparameter
ist $2c$, nicht $c$, weil die kumulantenerzeugende Funktion auch bei $t=-2c$
ausgewertet wird.

### `cmax_gauss(R)`

Soll $\sqrt{-\ln R}$ liefern — die Schranke ohne Formkorrektur, also für
symmetrisches $\Delta E$.

*Auffällig wäre:* etwas anderes als eine Zeile.

### `cmax_skew(R, g1, g2, c_hi=None, warn=True)`

Soll die **kleinste positive** Wurzel der Quartik unterhalb `c_hi` liefern
(Default `C_VALID`). Vorgehen: `numpy.roots` faktorisiert das Polynom
vollständig, dann Minimum der positiven reellen Wurzeln.

*Auffällig wäre:*

- ein Bracket-Verfahren (`brentq`) statt vollständiger Faktorisierung — das war
  eine frühere Fassung und scheitert bei $\gamma_2<0$, wo zwei positive Wurzeln
  existieren
- Rückfall auf `cmax_gauss`, wenn keine Wurzel existiert. Richtig ist der Rückfall
  auf `c_hi`, also die **obere** Grenze — eine zu tiefe Schranke würde den
  FAIL-Monitor zu früh feuern lassen
- fehlende Behandlung von $\gamma_2=0$ (dann ist es nur eine Kubik)

### `cmax_skew_vec(g1, g2, iters=8)`

Dasselbe für ganze Arrays, mit Newton ab der Gauß-Lösung, geklemmt auf
`C_VALID`. Existiert im Gültigkeitsbereich keine Wurzel, wird `C_VALID`
zurückgegeben.

*Auffällig wäre:* Rückfall auf die Gauß-Schranke statt auf `C_VALID` — dieselbe
Falle wie oben, nur in der vektorisierten Fassung.

### `log_neff_ratio(c, g1, g2)`

Soll $-c^2+\gamma_1c^3-\tfrac{7}{12}\gamma_2c^4$ liefern, also den Logarithmus
von $N_\text{eff}/n$ aus der Entwicklung.

*Anmerkung:* `reweighting.neff_ratio_cumulant` rechnet dasselbe. Zwei
Implementierungen derselben Formel sind eine Fehlerquelle. **?** Sollte
zusammengeführt werden.

### `running_moments` → `run_stats(dE)` / `run_stats_2d(D)`

Sollen nach jedem neuen Punkt $(k, c, \gamma_1, \gamma_2)$ liefern.
**Entscheidend: kausal.** Der Wert bei $k$ darf ausschließlich von `dE[:k]`
abhängen — hinge er auch von späteren Punkten ab, wäre die gesamte
Sequenzsimulation wertlos, weil der Monitor in die Zukunft schaute.

*Auffällig wäre:*

- eine Normierung über den vollen Datensatz statt über die ersten $k$ Punkte
  (etwa Mittelwert oder Standardabweichung von `dE` insgesamt)
- Unterschiede zwischen `run_stats` und `run_stats_2d` — sie sollen dasselbe
  rechnen, einmal für eine Sequenz, einmal zeilenweise für viele
- **?** `run_stats_2d` liefert unterhalb $k=4$ Rohwerte, `run_stats` NaN. Der
  Unterschied ist dokumentiert und unterhalb `K_FLOOR` bedeutungslos, aber es
  ist eine Inkonsistenz

### `se_c(c, g2, k)`

Soll $\mathrm{SE}(c) = c\sqrt{(\gamma_2+2)/4k}$ liefern — der Standardfehler der
Stichproben-Standardabweichung nach der Delta-Methode.

*Zum Einordnen:* $\gamma_2\ge-2$ ist eine Verteilungsschranke, als
Stichprobenwert kann es aber darunter rutschen. Der Ausdruck unter der Wurzel muss
abgesichert sein.

*Auffällig wäre:* ein Faktor 2 statt 4 im Nenner, oder $\gamma_2$ statt
$\gamma_2+2$ — beides plausible Tippfehler mit großer Wirkung.

### `stat_D(X)`

Soll je Zeile von `X` (Form $n\times k$) vier Größen liefern:
$(D, c, \gamma_1, \gamma_2)$ mit $D = c - c_\text{max}$.

*Auffällig wäre:*

- $c$ ohne $\beta$
- Momente über die falsche Achse (Zeilen statt Spalten)
- **?** `ddof`: intern wird auf $k-1$ korrigiert. Ob das mit `run_stats`
  übereinstimmt, weiß ich nicht sicher — das ist eine der Stellen, an denen ich
  genau hinschauen würde

### `checkpoint_grid(n, first_frac=0.10, ratio=1.4)`

Soll ein geometrisches Raster von $\lceil 0{,}1n\rceil$ bis $n$ liefern, Faktor
1,4. Die Absicht ist **Skalenfreiheit**: für $n=500$ und $n=5000$ sollen es
jeweils rund acht Blicke sein.

*Auffällig wäre:* ein festes Raster statt eines relativen — dann hinge die Zahl
der Blicke an $n$, und mit ihr die Mehrfachtest-Inflation.

### `monitor_split(D_seq, q=1.0, B=100, rng, chunk=250, checkpoints, k_floor)`

**Die verwendete Regel.** Soll je Zeile `(gefeuert, k_bei_dem_gefeuert)` liefern,
`k = -1` heißt „nie gefeuert".

Ablauf, wie er sein soll:

1. Checkpoints bestimmen, auf $\ge$ `k_floor` und $\le n$ beschränken
2. je Checkpoint $k$: nur die noch nicht gefeuerten Zeilen betrachten
3. $D$, $c$, $\gamma_2$ auf `D_seq[:, :k]` berechnen
4. $\mathrm{SE}(c_\text{max})$ aus `B` Bootstrap-Resamples **derselben $k$
   Punkte** — $c_\text{max}$ je Resample ist $c - D$
5. feuern, wo $D > q\,(\mathrm{SE}(c) + \mathrm{SE}(c_\text{max}))$, mit
   $\mathrm{SE}(c)$ analytisch aus `se_c`
6. gefeuerte Zeilen werden nicht weiter geprüft

*Auffällig wäre:*

- nur **ein** Band statt zwei — die alte Fassung `monitor_boot` bootstrappt $D$
  direkt. Beides ist verteidigbar, aber es sind verschiedene Regeln
- $\mathrm{SE}(c_\text{max})$ aus dem vollen Datensatz statt aus den ersten $k$
  Punkten — das wäre nicht kausal
- `q` mit Default 1.64 statt 1.0
- **?** Bei mehr als 250 Zeilen wird gechunkt, und der Zufallsstrom wird anders
  verbraucht. Ob das Ergebnis dadurch von `chunk` abhängt, weiß ich nicht

### `diagnose(R, g1, g2, rem_tol)`

Soll eine Liste von Klartext-Befunden liefern; leere Liste heißt „alle
Voraussetzungen erfüllt". Geprüft werden drei Dinge:

| Bedingung | wann verletzt |
|---|---|
| Eindeutigkeit der Wurzel | $\gamma_1^2 \ge \tfrac{56}{27}\gamma_2$ (und $\gamma_1>0$) |
| $N_\text{eff}\le n$ | $c_\text{max}$ jenseits der Stelle, wo die abgebrochene Reihe $N_\text{eff}>n$ behauptet |
| Restglied | $(2c_\text{max})^5/5! >$ `rem_tol` |

### `monitor_boot`, `decide_naive`, `fail_only_2d`

Frühere Fassungen der Regel. Werden vom Notebook **nicht mehr benutzt** — nur in
einem Kommentar erwähnt. Wenn die Erklärung sie als „die Regel" darstellt, ist das
veraltet.

---

## 3. `src/uq_mace/reweighting.py`

### `reweighting_weights(e_dft, e_model, beta)`

Soll $w_i = e^{-\beta(E_\text{DFT}-E_\text{model})_i}$ liefern.

*Wichtig:* $N_\text{eff}$ ist **invariant gegen einen konstanten Offset** in
$\Delta E$ — ein Offset ist ein gemeinsamer Faktor in allen $w_i$ und kürzt sich
im Kish-Quotienten heraus. Physikalisch ist das genau richtig, weil der absolute
Energienullpunkt bedeutungslos ist. Numerisch wird deshalb üblicherweise ein
Offset abgezogen, damit die Exponentiale nicht überlaufen.

*Auffällig wäre:* ein Vorzeichenfehler im Exponenten. Der ist heimtückisch, weil
$N_\text{eff}$ trotzdem plausibel aussieht — nur eben für die falsche Richtung
der Umgewichtung.

### `effective_sample_size(w)`

Soll $(\sum w)^2/\sum w^2$ liefern. Grenzfälle: alle $w$ gleich → $n$; ein $w$
dominiert alles → 1. Skaleninvariant in $w$.

### `psis_khat(w)`

Soll den Pareto-Tail-Index $\hat k$ liefern, geschätzt aus dem oberen Schwanz der
Gewichte. Bedeutung: $E[w^2]<\infty \iff \hat k<0{,}5$ — nur dann hat
$N_\text{eff}$ überhaupt einen Populationsgrenzwert.

*Zum Einordnen:* $\hat k$ ist ein **verrauschter** Schätzer, $\mathrm{SE}(\hat k)$
fällt nur wie $n^{-1/4}$. Bei einigen hundert Punkten streut er so stark, dass ein
einzelner Wert die 0,5-Schwelle nicht entscheiden kann. Das Gate gehört auf den
vollen Datensatz, nicht auf eine Teilstichprobe.

*Auffällig wäre:* eine feste Zahl von Schwanzpunkten statt einer $n$-abhängigen.

### `running_moments(x)`

Soll laufende Momente liefern, Grundlage von `run_stats`. Dieselbe
Kausalitätsanforderung wie dort.

*Auffällig wäre:* naive Summen von $x$, $x^2$, $x^3$, $x^4$ ohne Zentrierung —
das ist bei kleiner Varianz und großem Mittelwert numerisch heikel. **?** Ich weiß
nicht, welchen Algorithmus die Funktion benutzt.

### Nicht vom Workflow benutzt

`predicted_neff_gauss`, `khat_threshold`, `sample_overlap`, `reweighted_average`,
`running_neff_cv`, `neff_leave_one_out` — Diagnostik aus früheren Analysen.

`ensemble_shift` und `scale_to_system_size` rechnen die Übertragung vom Testsatz
auf einen Produktionslauf bzw. auf andere Systemgrößen. **Für das Notebook
gegenstandslos**, weil der Workflow auf dem Produktionsstrom selbst schätzt.

---

## 4. `src/uq_mace/predictions.py`

### `load_energies(path)`

Soll $(e_\text{DFT}, e_\text{MACE})$ aus einer npz-Datei liefern und **zwei
Formate** akzeptieren:

| Schlüssel im npz | Bedeutung |
|---|---|
| `e_dft` + `e_mace` | ältere Caches, `e_mace` ist bereits das Ensemble-Mittel |
| `e_dft` + `energies` | aktuelle Caches, `energies` hat eine Member-Achse, es wird gemittelt |

*Wichtig:* Beim zweiten Format wird über **Achse 0** gemittelt, also über die
Komitee-Member. Wäre es die falsche Achse, käme statt eines Ensemble-Mittels ein
Mittel über Frames heraus — und jede Zahl im Notebook wäre falsch, ohne dass
etwas offensichtlich kaputt aussähe.

Das ist die Funktion, an der buchstäblich alles hängt.

Die übrigen Funktionen (`load_weights`, `get_predictions`, `cache_path`,
`find_energy_cache`) verwaltet der Cache; das Notebook benutzt sie nicht.

---

## 4b. Wo der Code überhaupt liegt

Der zu prüfende Code verteilt sich auf **drei** Orte, nicht nur auf die
Bibliothek:

| Ort | Umfang | Testlage |
|---|---|---|
| `src/uq_mace/` | 31 öffentliche Funktionen | rund die Hälfte getestet |
| `notebooks/notebook_funktionen.py` | 20 Funktionen, 209 Zeilen | **kein Test** |
| `notebooks/notebook_zellen.py` | alle 25 Code-Zellen, davon **485 Zeilen Zellenkörper** | **kein Test** |

Der Zellenkörper ist mit 70 % des Notebook-Codes der größte Block — und der am
wenigsten beachtete. Dort stehen die Schleifen über $\rho$, die Etikettierung
„wahr PASS/FAIL", die Aggregation zu Prozentzahlen und die Ausgabe. **Genau dort
entstehen die Zahlen, die im Text stehen.** Ein Fehler dort — falsche
Vergleichsrichtung, falscher Nenner, ein Off-by-one beim Slicing — verfälscht
das Ergebnis, ohne dass die Bibliothek etwas falsch macht.

`notebook_zellen.py` enthält alle Zellen in Ausführungsreihenfolge mit
Abschnittsüberschrift. Die Datei ist nicht lauffähig (die Zellen bauen
aufeinander auf), sondern zum Lesen. Die Abhängigkeitsreihenfolge steht im Kopf
der Datei: Zelle 3 setzt Pfade und $\beta$, Zelle 6 die Ausreißermaske, Zelle 65
den Werkzeugblock, Zelle 73 den Sweep-Aufbau — danach ist alles verfügbar.

**Worauf im Zellenkörper besonders zu achten ist:**

| Zelle | Abschnitt | inline | was dort passiert |
|---|---|---|---|
| 67 | 7.2 | 52 | Sequenzen ziehen, Monitor laufen lassen, Urteil gegen die Wahrheit stellen, Ersparnis rechnen |
| 70 | 7.3 | 42 | Mutationsprobe, Abstand zur Entscheidungslinie |
| 6 | Setup | 41 | Ausreißer identifizieren und maskieren |
| 3 | Setup | 30 | $\beta$, Pfade, Modell-Dicts |
| 65 | 7.1 | 31 | Kontext setzen, Selbstprüfungen der Näherungen |
| 78 | 8.2 | 15 | $\rho$-Schleife Gauß gegen schief |
| 75, 81–83 | 8.1, 8.3 | 19–22 | die Messschleifen des Kapitels |

---

## 5. `notebooks/notebook_funktionen.py`

Die 20 Funktionen, die im Notebook selbst definiert sind — wortgleich extrahiert.
**Keine davon hat einen Test.** Die Datei nennt je Funktion Herkunft und freie
Variablen.

Sie sind so nicht lauffähig: sie ziehen `beta`, `R`, `CACHE`, `stat_D` und
anderes aus dem Notebook-Namensraum, statt sie als Argument zu bekommen.

### Die fünf, die Ergebnisse tragen

**`kish_ratio_rows(dE)`** — soll das exakte $N_\text{eff}/n$ je Zeile liefern,
**ohne jede Kumulantennäherung**. Das ist die *Wahrheit*, gegen die das
Monitor-Urteil bewertet wird. Ist sie falsch, sind alle Fehler- und
Erkennungsraten falsch. Sollte mit `effective_sample_size` übereinstimmen.

**`kish_truth(pool, N, seed)`** — dieselbe Wahrheit, auf einer großen Ziehung aus
`pool`. **?** Zieht als Offset `d.mean()` ab, während `kish_ratio_rows`
`dE.min(axis=1)` nimmt. Beides sollte dasselbe Ergebnis liefern (Offset-Invarianz),
aber es sind zwei Konventionen für dieselbe Größe — und wenn die Invarianz
irgendwo verletzt wäre, fiele es genau hier auf.

**`rescale_to(dE, c_target)`** — soll $\Delta E$ so skalieren, dass
$\beta\,\mathrm{std}=c_\text{target}$ gilt, **ohne $\gamma_1$ und $\gamma_2$ zu
verändern**. Eine reine Skalierung um den Mittelwert leistet das, weil Schiefe und
Kurtosis dimensionslos sind. Die ganze Grauzonen-Analyse steht darauf: sie
behauptet, die reale Form der Fehlerverteilung beizubehalten und nur den Abstand
zur Schranke zu variieren.

**`monitor_variante(...)`** — `monitor_split` mit wählbarer Schranke und
wählbarem Raster, für den Vergleich Gauß gegen schief. Mit Standardeinstellungen
muss es **identisch** zu `monitor_split` sein, sonst vergleicht die Analyse eine
andere Regel als der Workflow benutzt. Der Gauß-Zweig setzt bewusst kein Band um
die Schranke, weil sie dort nicht geschätzt, sondern fest ist.

**`feuert(sub, band_fn, ...)`** — ein einzelner Checkpoint mit wählbarer
Bandfunktion, für die Mutationsprobe. Soll bei der echten Bandfunktion dasselbe
liefern wie `monitor_split` an diesem einen Checkpoint.

### Die übrigen

`khat_rows`, `load_full`, `moments`, `se_line`, `expansion_terms`, `proof_stats`
tragen Nebenaussagen; `summary`, `moments_row`, `boot_band`, `print_table`,
`print_decomp`, `qq_row`, `kish_gauss_panel`, `proof_table`, `proof_bars` sind
Formatierung und Abbildungen.

Bei `load_full` eine Besonderheit: die Ausreißermaske `BAD` wird an **einem**
Modell bestimmt (dem L2-Referenzmodell) und auf alle angewandt. Begründung ist,
dass alle Modelle dieselben Strukturen sehen und die vier fehlerhaften Frames
DFT-seitig kaputt sind, nicht modellseitig.

---

## 6. Bereits geprüft — die Kippkandidaten (Stand 8. August)

Vorab durchgerechnet wurden die Fehler, die die Aussage *„der Workflow
funktioniert"* umwerfen würden. Alle sauber. Die Prüfungen sind reproduzierbar,
das Skript liegt als `notebooks/audit_kippkandidaten.py`.

| # | Prüfung | Vorgehen | Befund |
|---|---|---|---|
| K1 | **Lookahead** in `run_stats_2d` | `dE[k:]` komplett durch andere Werte ersetzen, Werte bei $k\le200$ vergleichen | sauber, Abweichung $\le2\cdot10^{-14}$ |
| K1b | **Lookahead** in `run_stats` | dito | sauber |
| K2 | **Lookahead** in `stat_D` | `stat_D(X[:, :k])` gegen `stat_D` auf den bereits gekürzten Daten | exakt gleich |
| K3 | **Lookahead** in `monitor_split`, ein Checkpoint | Zukunft ab $k$ ersetzen, Urteil bei $k$ vergleichen — bei $k=50,80,120$ | Urteile identisch |
| K4 | **Lookahead** über die ganze Checkpoint-Folge | Sequenz ab $k=214$ ersetzen; alle 34 Zeilen, die vorher feuern, müssen dasselbe $k$ behalten | identisch |
| K5 | Abhängigkeit von `chunk` | `chunk` = 50 / 100 / 250 / 1000, im Grenzbereich (Feuerrate 28 %) | keine Abweichung — numpy füllt denselben Zufallsstrom in derselben Reihenfolge |
| K6 | `rescale_to` erhält $\gamma_1,\gamma_2$ | Momente vor und nach der Skalierung auf drei Ziel-$c$ | exakt erhalten, Ziel-$c$ exakt getroffen |
| K7 | `load_energies`, Achsenwahl | synthetisches npz mit bekanntem Member-Mittel, beide Cache-Formate | über die Member-Achse gemittelt, beide Formate korrekt |
| K8 | Notebook-„Wahrheit" ≡ `effective_sample_size` | `kish_ratio_rows` gegen die Bibliotheksfunktion | identisch bis $4\cdot10^{-16}$ |
| K9 | Vorzeichen der Gewichte | Frame mit $E_\text{DFT}>E_\text{MACE}$ muss **kleineres** Gewicht bekommen | richtige Richtung |
| K10 | `ddof`-Konsistenz | $c,\gamma_1,\gamma_2$ bei $k=200$ aus `run_stats`, `run_stats_2d`, `stat_D` | alle drei `ddof=1`, identisch bis $5\cdot10^{-17}$ |

**Zu K1 im Besonderen.** `run_stats_2d` zieht in der ersten Zeile
`D.mean(axis=1)` ab, also den Mittelwert über *alle* $k$ — das sieht nach
Lookahead aus. Es ist eine Zentrierung, die sich in den Zentralmomenten wieder
herauskürzt; K1 bestätigt das numerisch. Die Zeile ist erklärungsbedürftig, aber
korrekt.

### Ein Befund, der keine Fehlfunktion ist

Im Grauzonen-Sweep ist „wahr PASS/FAIL" der **Populationswert**
($N=400\,000$ Ziehungen). Die einzelne 400-Punkt-Sequenz hat aber ihr eigenes
$N_\text{eff}/n$, und das streut:

| $\rho$ | Population | Label | Anteil Sequenzen, die **selbst** unter $R$ liegen |
|---|---|---|---|
| 0,80 | 0,863 | PASS | 0,0 % |
| 0,90 | 0,833 | PASS | 0,4 % |
| 0,95 | 0,817 | PASS | **7,5 %** |
| 1,00 | 0,801 | PASS | **42,4 %** |
| 1,05 | 0,785 | FAIL | 86,1 % |

Bei $\rho=0{,}95$ sind also 7,5 % der als PASS gelabelten Sequenzen tatsächlich
unter $R$ — ein Teil der dort gezählten „Fehlalarme" ist keiner. Die Regel ist
in der Grauzone damit eher besser als die Zahlen nahelegen. Wo die belastbaren
Aussagen liegen ($\rho\le0{,}90$), ist der Effekt mit 0,4 % ohne Bedeutung.

Das ist eine Frage der Etikettierung, nicht des Codes: die
Populations-Wahrheit ist die richtige Referenz für „taugt das Modell", die
Sequenz-Wahrheit für „war der Abbruch dieser Rechnung richtig".

### Was diese Vorprüfung **nicht** abgedeckt hat

- **Herleitung von `se_c`** — die Formel wird benutzt, aber nicht gegen die
  Theorie verifiziert. Geprüft ist nur, dass sie den Bootstrap derselben
  Stichprobe trifft
- **Kalibrierung von `psis_khat`** — nie gegen eine GPD mit bekanntem $k$ gehalten
- **Seed-Empfindlichkeit von `kish_truth`** — streut der Wert über Seeds weniger,
  als die berichtete Genauigkeit suggeriert?
- **`load_full`** — die Ausreißermaske wird an einem Modell bestimmt und auf alle
  angewandt

Keiner dieser Punkte würde die Grundaussage kippen, aber sie sind offen.

---

## 7. Wo nichts geprüft ist

Nach Gewicht sortiert — dort würde ich am genauesten hinhören:

| | ohne Test |
|---|---|
| 1 | `load_energies` — Wurzel aller Zahlen, zwei Formate, Achsenwahl |
| 2 | `kish_ratio_rows`, `kish_truth` — die Wahrheit, gegen die bewertet wird |
| 3 | `run_stats`, `run_stats_2d`, `running_moments` — Kausalität |
| 4 | `monitor_variante` — muss `monitor_split` entsprechen |
| 5 | `rescale_to` — Invarianz von $\gamma_1,\gamma_2$ |
| 6 | `stat_D` — nur indirekt geprüft, `ddof`-Konvention offen |
| 7 | `se_c` — nur indirekt geprüft |

Gut abgedeckt sind `cmax_skew` (15 Tests), `monitor_split` (10),
`cmax_skew_vec` (5), `diagnose` (5), `cmax_gauss` (5).

**Nebenbei:** `tests/test_calibration.py` importiert `uq_mace.calibration`, und
das Modul existiert nicht. Der Import bricht die Test-Collection ab — ohne
`--ignore` laufen also gar keine Tests. Testlauf:
`PYTHONPATH=src pytest tests/ -q --ignore=tests/test_calibration.py`

---

## 8. Was eine Code-Durchsicht nicht klären kann

Damit die Erwartung stimmt:

- **Die Testpunkte stammen aus verschiedenen MD-Läufen.** Validiert wird die
  Methodik des iterativen Prüfens, nicht die Güte eines konkreten L0- oder
  L2c-Modells. Eine absolute Modellaussage wäre ohnehin nicht möglich, weil die
  Performance stark von Temperatur und Modellgröße abhängt.
- **Autokorrelation.** Alle Sequenzen sind iid gezogen; ein realer MD-Strom ist
  korreliert, dann wäre $\mathrm{SE}(c)$ um $\sqrt\tau$ größer.
- **Abdeckung.** Kein Screening kann ausschließen, dass ein wichtiger Bereich des
  Konfigurationsraums unbesucht bleibt.
- **Die Grauzone um $\rho=1$** ist Stetigkeit der Gütefunktion, keine
  Implementierungsschwäche: wer $\rho=1{,}05$ erkennt, muss bei $\rho=0{,}95$
  gelegentlich feuern.
- **$\gamma_1<0$ ist ungeprüft.** Kein reales Modell hat Linksschiefe. Das
  Vorzeichen entscheidet aber, in welche Richtung die Gauß-Schranke irrt — bei
  Rechtsschiefe ist sie konservativ, bei Linksschiefe nicht.
