# kish_screening.py

Beantwortet für einen Satz DFT- und ML-Energien eine Frage:

> **Wird thermodynamisches Reweighting auf diesen Daten statistisch tragen — und
> hätte man den teuren Teil der Rechnung früher abbrechen können?**

Eine Datei, nur numpy. Version 1.1.

---

## 1. Die Größe, um die es geht

Reweighting korrigiert Mittelwerte eines ML-Potentials auf DFT-Niveau:

$$\langle A\rangle_\text{DFT} \approx \frac{\sum_i A_i w_i}{\sum_i w_i},
\qquad w_i = e^{-\beta(E_\text{DFT}(R_i)-E_\text{ML}(R_i))}$$

Ob das etwas taugt, hängt daran, wie ungleich die Gewichte werden. Das misst der
**Kish-Wert**

$$\frac{N_\text{eff}}{n} = \frac{(\sum_i w_i)^2}{n\sum_i w_i^2}\ \in (0,1].$$

Bei $N_\text{eff}/n = 0{,}9$ verhält sich die umgewichtete Stichprobe wie 90 %
unabhängiger Punkte. Das Kriterium $N_\text{eff}/n \ge R$ ist **skalenfrei** —
mehr Frames helfen nicht, wenn die Gewichtsverteilung schlecht ist.

---

## 2. Der Formalismus

Mit $\Delta E = \mu + \sigma z$ wird $w = e^{-\beta\mu}\cdot e^{-cz}$. Der Offset
kürzt sich heraus, übrig bleibt eine einzige Skala:

$$\boxed{c = \beta\,\mathrm{std}(\Delta E)}$$

Über die kumulantenerzeugende Funktion folgt exakt

$$\log\frac{N_\text{eff}}{n} = 2K(-c)-K(-2c)
= -c^2 + \gamma_1c^3 - \tfrac{7}{12}\gamma_2c^4 + O(c^5).$$

Effektiver Entwicklungsparameter ist $2c$, nicht $c$ — $K$ wird auch bei $t=-2c$
ausgewertet. Aus der Restabschätzung $(2c)^5/5! \le 0{,}05$ folgt die
Gültigkeitsgrenze $C_\text{VALID} = \tfrac12(120\cdot0{,}05)^{1/5} \approx 0{,}7155$.

Aufgelöst nach $c$ gibt das Effizienzkriterium die **Schranke** $c_\text{max}$:
die kleinste positive Wurzel von

$$c^2-\gamma_1c^3+\tfrac{7}{12}\gamma_2c^4 = -\ln R,$$

gegen die Gauß-Variante $\sqrt{-\ln R}$, die die Form ignoriert. Rechtsschiefe
erlaubt mehr $c$; die Gauß-Schranke ist dann konservativ, bei Linksschiefe kehrt
sich das um.

Entschieden wird über das Verhältnis der beiden:

$$\boxed{\rho = \frac{c}{c_\text{max}}}\qquad
\rho \le 1 \Rightarrow \text{PASS},\qquad \rho > 1 \Rightarrow \text{FAIL}$$

$\rho$ ist die eigentliche Kennzahl der Ausgabe. Sie macht Modelle vergleichbar,
deren $c_\text{max}$ wegen unterschiedlicher Schiefe verschieden ausfällt: $c$
allein sagt nichts, solange die Schranke nicht danebensteht.

### Die drei Größen und ihre Rollen

| Größe | Rolle | Konvergenz |
|---|---|---|
| $c$ | Skala, Entscheidungsgröße | stabil ab $k\approx14$; $\mathrm{SE}/c=\sqrt{(\gamma_2+2)/4k}$ |
| $\gamma_1,\gamma_2$ | Formkorrektur, bestimmen $c_\text{max}$ | $\gamma_1$ stabil erst ab $k\approx210$ |
| $\hat k$ (Pareto-Tail) | **Existenzbedingung**, Gate — keine Prognose | $\mathrm{SE}\propto n^{-1/4}$ |

**$c$ allein ist kein Prädiktor.** Bei festem $c$ lässt sich $N_\text{eff}/n$
zwischen 0,0007 und 0,99998 konstruieren. Verteilungsfrei gilt nur
$c\to0 \Rightarrow N_\text{eff}/n\to1$.

**$\hat k$ ist ein Gate, kein Qualitätsmaß.** $E[w^2]<\infty \iff \hat k<0{,}5$.
Ist das verletzt, ist $K(-2c)$ undefiniert und die Herleitung nicht ungenau,
sondern gegenstandslos. $\hat k$ ist nicht extrapolierbar — es gibt keine
Funktion $\hat k(c)$.

---

## 3. Die drei $N_\text{eff}$-Zeilen

| Zeile | Herkunft | Rolle |
|---|---|---|
| **exakt** | $(\sum w)^2/(n\sum w^2)$ über alle $n$ Gewichte | annahmefrei; **allein diese Zahl entscheidet** |
| Reihe | $\exp(-c^2+\gamma_1c^3-\tfrac{7}{12}\gamma_2c^4)$ | Diagnose: trägt die Entwicklung bei diesem $c$? |
| Gauss allein | $\exp(-c^2)$ | zeigt, wie viel die Formkorrektur ausmacht |

Weichen „exakt" und „Reihe" deutlich ab, hält die Entwicklung nicht mehr — dann
ist $c_\text{max}$ unbrauchbar, der exakte Kish-Wert aber weiterhin gültig. Die
Kumulantenreihe geht **nicht** in das gemeldete $N_\text{eff}/n$ ein; sie liefert
über die Quartik nur die Schranke, gegen die der Monitor vergleicht.

Das gemeldete Restglied $(2c)^5/5! > 0{,}05$ und $c > C_\text{VALID}$ sind
dieselbe Bedingung, nur anders geschrieben.

---

## 4. Der sequenzielle Monitor

Während die Gewichte nach und nach anfallen, wird an einem geometrischen Raster
geprüft:

$$\hat c(k) - \mathrm{SE}\big(\hat c(k)\big)
\;>\;
\hat c_\text{max}(k) + \mathrm{SE}\big(\hat c_\text{max}(k)\big)
\;\Longrightarrow\; \text{FAIL, abbrechen}$$

Drei Eigenschaften, die die Form erklären:

**Einseitig.** Ein frühes PASS spart nichts — die Gewichte werden am Ende ohnehin
vollständig gebraucht. Nur ein früh abgesichertes FAIL spart Rechenzeit. Damit
bleibt genau eine Fehlerart: ein brauchbares Modell abbrechen.

**Beide Seiten sind laufende Schätzungen.** $\hat c_\text{max}$ streut 30–60 %
stärker als $\hat c$; ein Band nur um $\hat c$ ließe die größere der beiden
Unsicherheiten weg.

**Ein Standardfehler je Seite, kein Vorfaktor.** Bei zwei verglichenen Bändern
wäre ein $q$ kein Niveau — nichtüberlappende Konfidenzbänder sind ein
konservativer Test für eine Differenz, ein nominelles 5 % wäre effektiv etwa
0,5 %. Statt einer Niveauaussage, die nicht hält, ist die Bandbreite eine
**Konvention**.

Alle Größen bei $k$ benutzen ausschließlich die ersten $k$ Punkte — der Monitor
sieht die Zukunft nicht. Das Raster beginnt bei `first_frac·n` (Default 10 %),
wächst mit Faktor 1,4, und `k_floor` (Default 50) schneidet alles darunter ab.
Beides sind **getrennte** Parameter: für $n \ge 500$ liegt der Rasteranfang über
50 und der Filter ist wirkungslos, darunter greift er.

**Warum nicht früher geschaut wird.** Der späte Start ist der eigentliche Hebel,
nicht die Rasterdichte: von $k\ge5$ auf $k\ge50$ fällt der Fehlalarm um Faktor 6,
obwohl sich die Zahl der Blicke nur um 11 % verringert. Bei $k=5$ hat die Quartik
in 43 % der Ziehungen im Gültigkeitsbereich gar keine Wurzel — die Schranke ist
dort nicht ungenau, sondern nicht definiert.

---

## 5. Konventionen

### ddof

$c$ wird mit `ddof=1` gebildet, $\gamma_1$ und $\gamma_2$ mit `ddof=0`. Das ist
keine Frage der Bibliothek — `ddof` ist eine statistische Wahl, und numpys
eigener Default ist selbst `ddof=0`. Der Grund für die Mischung:

* **$c$ mit `ddof=1`.** Ein Skalenparameter, den man erwartungstreu schätzen will
  — Bessel-Korrektur, beim zweiten Moment exakt.
* **$\gamma_1,\gamma_2$ als Plug-in.** Die Kumulantenentwicklung ist in
  *Populations*kumulanten formuliert; die natürlichen Stichprobenanaloga sind
  $m_3/m_2^{3/2}$ und $m_4/m_2^2-3$. Biaskorrigierte Varianten (Fishers $G_1$,
  $G_2$) schätzen etwas anderes.

Zur Gegenprobe von außen entsprechen $\gamma_1,\gamma_2$ exakt
`scipy.stats.skew`/`kurtosis` mit `bias=True`; das Skript benutzt scipy nicht.
Der Unterschied zwischen den Konventionen ist klein gegen $\mathrm{SE}(c)$ — bei
$n=400$ 0,13 % gegen 3,86 % — aber nicht verschwindend: am ersten Checkpoint bei
$k=50$ beträgt er 1 %, was an der Entscheidungslinie gelegentlich ausreicht.

### Gewichte

$w = e^{-\beta(\Delta E - \min\Delta E)}$. Der Abzug macht den größten Exponenten
exakt null: Überlauf ist ausgeschlossen, möglich bleibt nur Unterlauf der ohnehin
vernachlässigbaren Gewichte. $N_\text{eff}$ ist gegen einen konstanten Offset in
$\Delta E$ invariant, das Ergebnis ändert sich also nicht.

### Die zwei Standardfehler

Sie kommen aus verschiedenen Quellen, und das ist gemessen, nicht gesetzt:

| | Herkunft | Begründung |
|---|---|---|
| $\mathrm{SE}(\hat c)$ | analytisch, $\hat c\sqrt{(\hat\gamma_2+2)/4k}$ | trifft den Bootstrap derselben Sequenz auf 3 % |
| $\mathrm{SE}(\hat c_\text{max})$ | gebootstrappt | die analoge Delta-Methode liegt im Mittel 60 % zu hoch, bei $k=50$ Ausreißer bis Faktor 35 — $f'(c_\text{max})$ steht im Nenner und geht bei verrauschtem $\hat\gamma_1$ gegen null |

Der Bootstrap ist über `--seed` reproduzierbar. Die geschätzte Bandbreite streut
selbst um $1/\sqrt{2B}$ — bei $B=200$ sind das 5 %. Wer knapp an der
Entscheidungsgrenze liegt, nimmt `-B 1000`.

### Eingabekonventionen

**Energien pro Zelle, nicht pro Atom.** Das Kriterium hängt an der Streuung der
*Gesamtenergie*; wegen $c\propto\sqrt N$ liefern Energien pro Atom stillschweigend
ein falsches $c$. Das Skript kann das nicht bemerken.

**Reihenfolge = Anfallreihenfolge.** Der Monitor liest die Zeilen als den Strom,
in dem die Punkte entstehen.

**Einheiten** über `-u` (Default eV), **Temperatur** über `-T`. Beide gehen über
$\beta = 1/k_BT$ direkt in $c$ ein.

---

## 6. Grenzen

**Die Grauzone um $\rho = 1$ lässt sich nicht wegkalibrieren.** Zur Erinnerung:
$\rho = c/c_\text{max}$, die Grenze liegt bei 1. $\rho=0{,}99$ und $\rho=1{,}01$
unterscheiden sich in $N_\text{eff}/n$ um 0,003. Jede Regel, die $\rho=1{,}05$
erkennt, muss bei $\rho=0{,}95$ gelegentlich feuern — Stetigkeit der
Gütefunktion, kein Umsetzungsfehler.

| Regime | $\rho$ | Verhalten | belastbar? |
|---|---|---|---|
| sicher PASS | $\lesssim0{,}90$ | Fehlalarm $\le0{,}2$ % | **ja** |
| Grauzone | $0{,}95\dots1{,}2$ | Fehlalarm bis 11 %, Erkennung erst 76 % bei $\rho=1{,}1$ | **nein** |
| sicher FAIL | $\gtrsim1{,}2$ | Erkennung 99 %, Abbruch beim ersten Checkpoint | **ja** |

**Korrelierte Daten machen den Monitor zu selbstsicher.** Die Standardfehler
setzen unabhängige Punkte voraus; $\mathrm{SE}(c)$ kennt keine
Autokorrelationszeit. Auf einem AR(1)-Strom mit $\rho_1 = 0{,}38$ gegen
iid-Ziehungen aus derselben Randverteilung:

| $\rho$ (wahr PASS) | zusammenhängende Fenster | iid |
|---|---|---|
| 0,90 | 0 % | 0 % |
| 0,95 | **6 %** | 1 % |

Aus $\tau = (1+\rho_1)/(1-\rho_1) \approx 2{,}2$ folgt $\sqrt\tau \approx 1{,}5$:
die Bänder müssten rund 50 % breiter sein. Eine Korrektur ist **nicht** eingebaut.
Bei dicht aufeinanderfolgenden Frames ausdünnen oder das Ergebnis als optimistisch
lesen.

**$\hat k$ ist stark verrauscht.** Der Standardfehler fällt nur wie $n^{-1/4}$ und
liegt bei $n=500$ bei etwa 0,15 — ein einzelner Wert kann die 0,5-Schwelle nicht
sicher entscheiden. Das Skript rechnet $\hat k$ deshalb immer auf dem **vollen**
übergebenen Satz, nie auf einem Präfix. In einem Produktionslauf gehört das Gate
auf den vollständigen Testsatz, nicht auf die laufende Sequenz.

**$N_\text{eff}$ misst Ungleichheit, nicht Abdeckung.** Gleichmäßige Gewichte
schließen nicht aus, dass eine wichtige Region des Konfigurationsraums nie besucht
wurde.

**Reweighting korrigiert auf die DFT-Referenz, nicht auf die Realität.** Fehler
des Funktionals bleiben unangetastet.

---

## 7. Aufruf

```bash
python3 kish_screening.py DFT_DATEI [ML_DATEI] [Optionen]
```

Welcher Modus greift, entscheidet allein die Zahl der Positionsargumente. Mit
**einer** Datei muss eine npz mit `e_dft` und `e_mace`/`energies` vorliegen
(bei `energies` der Form $(M,F)$ wird über Achse 0 gemittelt); `--key-*` wirkt
dann nicht. Mit **zwei** Dateien werden `.npy .npz .txt .dat .csv .tsv` gelesen,
Schlüssel über `--key-dft` / `--key-ml`.

| Option | Bedeutung | Default |
|---|---|---|
| `-R` | gefordertes $N_\text{eff}/n$ | 0.8 |
| `-T` | Temperatur in Kelvin | 292 |
| `-u` | Einheit der Eingabeenergien | eV |
| `-k` | Checkpoints unterhalb $k$ verwerfen | 50 |
| `--first-frac` | Rasteranfang als Anteil von $n$ | 0.10 |
| `-b` | Bandbreite in Standardfehlern je Seite | 1.0 |
| `-B` | Bootstrap-Resamples je Checkpoint | 200 |
| `--seed` | Zufallsstartwert | 0 |

---

## 8. Ergebnisse exportieren

Vier Ausgabeformen, alle auf **stdout**; Warnungen gehen getrennt auf **stderr**
und lassen sich mit `2>/dev/null` unterdrücken, ohne das Ergebnis zu verlieren.

| Form | Aufruf | Inhalt |
|---|---|---|
| Bericht | (Default) | formatierte Kennzahlen und Urteil |
| + Checkpoints | `--steps` | zusätzlich die Tabelle aller Blicke des Monitors |
| Kurzform | `-q` | eine Zeile: `PASS`, `FAIL` oder `UNKLAR` |
| Maschinenlesbar | `--json` | vollständige Struktur, siehe unten |
| ohne Monitor | `--no-monitor` | nur die Kennzahlen des vollen Satzes |

### JSON-Struktur

```
gesamt/
  n, T, beta, R, units, band          Eingabeparameter
  sigma, c, gamma1, gamma2            Momente von dE
  c_max, c_max_gauss, rho             Schranke und Kennzahl
  neff_ratio                          exakter Kish-Wert  <- entscheidet
  neff_ratio_reihe, neff_ratio_gauss  Vergleichswerte (Diagnose)
  khat, r5                            Gate und Restglied
  k_floor, first_frac                 Rasterparameter
  diagnose[]                          Meldungen zur Quartik
version                               "1.1"
hinweise[], warnungen[]               Klartextmeldungen
monitor/
  gefeuert                            true/false
  k_stop                              Abbruchpunkt oder null
  gespart                             Anteil eingesparter DFT-Punkte
  checkpoints[]                       das benutzte Raster
  schritte[]                          je Checkpoint: k, c, gamma1, gamma2,
                                      c_max, se_c, se_c_max, abstand, band, feuert
urteil                                "PASS" | "FAIL" | "UNKLAR"
begruendung                           Klartext
```

`schritte[]` enthält $\mathrm{SE}(\hat c)$ und $\mathrm{SE}(\hat c_\text{max})$
**getrennt** — im Bericht steht nur ihre Summe unter „Band". Für die Frage, ob
die Schranke oder die Skala die Unsicherheit dominiert, ist die Aufschlüsselung
nötig.

### Exit-Codes

| Code | Bedeutung |
|---|---|
| 0 | PASS |
| 1 | FAIL |
| 2 | Aufrufsfehler |
| 3 | Datenfehler |
| 4 | UNKLAR — $\hat k \ge 0{,}5$, $c \ge C_\text{VALID}$, oder die Reihe behauptet an $c_\text{max}$ selbst $N_\text{eff} > n$ |

UNKLAR wird nur gemeldet, wenn der exakte Kish-Wert nicht ohnehin FAIL sagt — ein
FAIL aus der annahmefreien Zahl steht unabhängig von jeder Reihenentwicklung.

### Beispiele

```bash
# Kennzahl in eine Variable
rho=$(python3 kish_screening.py daten.npz --json --no-monitor 2>/dev/null \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['gesamt']['rho'])")

# Verlauf der Bänder über die Checkpoints als CSV
python3 kish_screening.py daten.npz --json 2>/dev/null | python3 -c "
import json,sys
print('k,c,c_max,se_c,se_c_max,feuert')
for s in json.load(sys.stdin)['monitor']['schritte']:
    print(f\"{s['k']},{s['c']:.6f},{s['c_max']:.6f},{s['se_c']:.6f},{s['se_c_max']:.6f},{s['feuert']}\")
" > verlauf.csv

# Als Bedingung im Workflow
python3 kish_screening.py daten.npz -q >/dev/null 2>&1 || exit 1
```

---

## 9. Methodik

Der Formalismus ist die auf Modellfehler übertragene Form eines etablierten
Kriteriums: $\beta\sigma_{\Delta U}$ ist in der Freie-Energie-Störungstheorie
seit Zwanzig die Kenngröße für Konvergenz, mit der Faustregel
$\beta\sigma \lesssim 1$. Wu & Kofke (JCP 123, 054103 und 084109, 2005) bauen
darauf ein Bias-Maß über den Phasenraum-Überlapp, allerdings unter Gauß-Annahme —
die $\gamma_1c^3$-Korrektur schließt genau diese Lücke. Kofke (JCP 117, 6911,
2002) leitet strukturell dieselbe Bedingung für die Akzeptanzrate im Replica
Exchange her.

Der $\hat k$-Schätzer folgt Zhang & Stephens (2009) in der Form, die Vehtari
et al. (JMLR 25, 2024) für PSIS benutzen.

Anwendungskontext: Hilpert & Kresse, *Accurate thermophysical properties of water
using machine-learned potentials*, J. Chem. Phys. 164, 194504 (2026).

Die Kernfunktionen sind gegen `uq_mace.screening` / `uq_mace.reweighting`
geprüft: bitgleich bei `momente`, `se_c`, `gewichte`, `neff_ratio`, `psis_khat`,
`log_neff_ratio`, `diagnose`; `cmax_skew` gegen die Newton-Fassung
$4{,}7\cdot10^{-15}$ auf 3721 Parameterpaaren; 90 von 90 Monitorläufen
entscheidungsgleich.