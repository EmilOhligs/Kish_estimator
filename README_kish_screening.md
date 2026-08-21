# kish_screening.py

Beantwortet für einen Satz DFT- und ML-Energien eine Frage:

> **Wird thermodynamisches Reweighting auf diesen Daten statistisch tragen — und
> hätte man den teuren Teil der Rechnung früher abbrechen können?**

Eine Datei, nur numpy, über die Kommandozeile bedienbar, mit Exit-Codes für die
Verwendung in Shell-Skripten.

---

## 1. Wozu

Reweighting korrigiert Mittelwerte eines ML-Potentials auf DFT-Niveau:

$$\langle A\rangle_\text{DFT} \approx \frac{\sum_i A_i w_i}{\sum_i w_i},
\qquad w_i = e^{-\beta(E_\text{DFT}(R_i)-E_\text{ML}(R_i))}$$

Ob das etwas taugt, hängt daran, wie ungleich die Gewichte werden. Das misst der
**Kish-Wert**

$$\frac{N_\text{eff}}{n} = \frac{(\sum_i w_i)^2}{n\sum_i w_i^2}\ \in (0,1].$$

Bei $N_\text{eff}/n = 0{,}9$ verhält sich die umgewichtete Stichprobe wie 90 %
unabhängiger Punkte. Bei 0,2 ist die Rechnung praktisch wertlos, egal wie viele
Frames man noch anhängt — das Kriterium ist **skalenfrei**, mehr Punkte helfen
nicht.

Die DFT-Rechnungen sind der teure Teil. Deshalb zwei Ausgaben:

1. **Das Urteil auf dem vollen Satz** — taugt das Modell?
2. **Die Simulation des sequenziellen Monitors** — nach wie vielen Punkten hätte
   man das schon gewusst und abbrechen können?

---

## 2. Voraussetzungen

Python 3.9 oder neuer und **numpy**. Sonst nichts.

```bash
python3 -c "import numpy; print(numpy.__version__)"   # muss durchlaufen
chmod +x kish_screening.py                            # optional
```

---

## 3. Eingabedaten

Zwei Dateien: **DFT-Energien** und **ML-Energien**, gleiche Länge, **gleiche
Reihenfolge** — Zeile $i$ beider Dateien muss denselben Frame beschreiben. Das
prüft das Skript nicht; es kann nur die Länge vergleichen.

| Endung | Format |
|---|---|
| `.npy` | numpy-Array, 1D |
| `.npz` | Schlüssel `e_dft`, `e_mace`, `e_model`, `energies`, `energy`, `E` oder `e`; sonst `--key-dft` / `--key-ml` angeben |
| `.txt` `.dat` | eine Zahl pro Zeile, `#` `!` `%` sind Kommentar |
| `.csv` `.tsv` | dito, kommagetrennt |

**Zwei Achsen** in einem npz (etwa ein Komitee mehrerer Modelle) werden als
`(member, frame)` gelesen und über die Member gemittelt.

**Einheiten:** Default eV. Mit `-u` umstellen — `meV`, `Ha`, `Ry`, `kcal/mol`,
`kJ/mol`. Das ist keine Kosmetik: die Temperatur geht über $\beta = 1/k_BT$ ein,
eine falsche Einheit verfälscht jedes Ergebnis.

**Energien pro Zelle, nicht pro Atom.** Das Kriterium hängt an der Streuung der
*Gesamtenergie* des simulierten Systems.

---

## 4. Aufruf

```bash
python3 kish_screening.py DFT_DATEI ML_DATEI [Optionen]
```

| Option | Bedeutung | Default |
|---|---|---|
| `-R`, `--target` | gefordertes $N_\text{eff}/n$, in $(0,1)$ | 0.8 |
| `-T`, `--temperature` | Temperatur in Kelvin | 292 |
| `-u`, `--units` | Einheit der Eingabeenergien | eV |
| `--key-dft`, `--key-ml` | npz-Schlüssel | automatisch |
| `-k`, `--k-floor` | erster Blick des Monitors | max(50, n/10) |
| `-b`, `--band` | Bandbreite in Standardfehlern je Seite | 1.0 |
| `-B`, `--bootstrap` | Resamples je Checkpoint | 200 |
| `--seed` | Zufallsstartwert | 0 |
| `--no-monitor` | nur Kennzahlen, keine Simulation | |
| `--steps` | Tabelle aller Checkpoints | |
| `--json` | Ergebnis als JSON auf stdout | |
| `-q`, `--quiet` | nur `PASS` / `FAIL` / `UNKLAR` | |

---

## 5. Exit-Codes

| Code | Bedeutung |
|---|---|
| **0** | **PASS** — Kriterium erfüllt, Reweighting trägt |
| **1** | **FAIL** — Abbruchbedingung erfüllt, Rechnung einstellen |
| **2** | Aufrufsfehler: Argument unzulässig, Datei fehlt, Format unbekannt |
| **3** | Datenfehler: ungleiche Länge, zu wenige Punkte, NaN, konstantes $\Delta E$ |
| **4** | **UNKLAR** — weder PASS noch FAIL; die Voraussetzungen der Methode sind verletzt (siehe §7) |

Meldungen und Warnungen gehen auf **stderr**, das Ergebnis auf **stdout**. Beides
lässt sich getrennt umleiten.

### Verwendung in Shell-Skripten

```bash
# Einfachster Fall: abbrechen, wenn das Modell nicht taugt
python3 kish_screening.py e_dft.npy e_ml.npy || {
    echo "Reweighting traegt nicht — MD stoppen" >&2
    exit 1
}

# Alle Faelle unterscheiden
python3 kish_screening.py e_dft.npy e_ml.npy -q
case $? in
    0) echo "weiter" ;;
    1) echo "abbrechen" ;;
    4) echo "nicht belastbar — von Hand ansehen" ;;
    *) echo "Aufruf- oder Datenfehler" ;;
esac

# Als Bedingung, ohne Ausgabe
if python3 kish_screening.py a.npy b.npy -q >/dev/null 2>&1; then
    echo "traegt"
fi

# Kennzahl weiterreichen
python3 kish_screening.py a.npy b.npy --json --no-monitor \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['gesamt']['neff_ratio'])"

# Mehrere Modelle vergleichen
for m in modell_*.npy; do
    printf '%-24s %s\n' "$m" "$(python3 kish_screening.py dft.npy "$m" -q --no-monitor 2>/dev/null)"
done
```

---

## 6. Die Ausgabe lesen

```
  std(dE)             8.220 meV
  c = beta*std(dE)    0.3267
  Schiefe   gamma1    +0.5027
  Kurtosis  gamma2    +0.4071

  c_max (Gauss)       0.4724
  c_max (schief)      0.5279   <- verwendet
  rho = c/c_max       0.619

  N_eff/n (exakt)     0.9125   >= R = 0.8
  khat (Tail-Index)   +0.068   Gate bestanden
  Restglied (2c)^5/5! 0.0010   ok
```

**$c = \beta\,\mathrm{std}(\Delta E)$** ist die Entscheidungsgröße — die Breite
der Fehlerverteilung in Einheiten der thermischen Energie. Sie allein sagt noch
nichts: bei festem $c$ lässt sich $N_\text{eff}/n$ zwischen 0,0007 und 0,99998
konstruieren. Es braucht die Form dazu.

**$\gamma_1$, $\gamma_2$** sind Schiefe und Exzess-Kurtosis von $\Delta E$. Sie
sind nicht Beiwerk, sondern die Taylorkoeffizienten der Entwicklung

$$\log\frac{N_\text{eff}}{n} = -c^2 + \gamma_1c^3 - \tfrac{7}{12}\gamma_2c^4 + O(c^5).$$

**$c_\text{max}$** ist die Schranke, ab der das Ziel $R$ verfehlt wird: die
kleinste positive Wurzel von $c^2-\gamma_1c^3+\tfrac{7}{12}\gamma_2c^4 = -\ln R$.
Die Gauß-Variante $\sqrt{-\ln R}$ steht zum Vergleich daneben — sie ignoriert die
Form und liegt bei Rechtsschiefe zu tief. Verwendet wird immer die
schiefe-korrigierte.

**$\rho = c/c_\text{max}$** ist die eigentliche Kennzahl. $\rho\le1$ heißt PASS.
Unterhalb 0,9 und oberhalb 1,2 ist die Aussage belastbar, dazwischen liegt eine
Grauzone (siehe §7).

**$N_\text{eff}/n$** ist der exakte Kish-Wert, ohne jede Näherung — die
verlässlichste Zahl der ganzen Ausgabe.

**$\hat k$** ist der Pareto-Tail-Index. Er ist kein Qualitätsmaß, sondern ein
**Gate**: nur für $\hat k < 0{,}5$ existiert $E[w^2]$ überhaupt, und nur dann hat
$N_\text{eff}$ einen Grenzwert, gegen den es konvergieren könnte.

**Restglied** $(2c)^5/5!$ schätzt den Abbruchfehler der Reihe. Über 0,05 trägt
sie nicht mehr — dann ist $c_\text{max}$ unbrauchbar, der exakte Kish-Wert aber
weiterhin gültig.

### Der Monitor

```
  Regel: c(k) - 1*SE(c) > c_max(k) + 1*SE(c_max)
  Checkpoints: [50, 70, 98, 137, 192, 269, 377, 400]

        k        c    c_max   Abstand     Band    Urteil
       50   1.1223   0.4892   +0.6332   0.1491      FAIL

  -> Abbruch bei k = 50 von 400
     88 % der DFT-Punkte waeren gespart worden.
```

Der Monitor geht die ersten $k$ Punkte durch — **in der Reihenfolge, in der sie
in der Datei stehen** — und prüft an einem geometrischen Raster, ob die Bänder um
$\hat c$ und $\hat c_\text{max}$ sich getrennt haben. Erst dann wird FAIL
behauptet.

Die Regel ist **einseitig**: PASS wird nie früh behauptet. Ein frühes PASS spart
nichts, weil die Gewichte am Ende ohnehin vollständig gebraucht werden — nur ein
frühes FAIL spart Rechenzeit. Damit gibt es genau eine Fehlerart: ein brauchbares
Modell abbrechen.

Alle Größen bei $k$ benutzen ausschließlich die ersten $k$ Punkte. Der Monitor
sieht die Zukunft nicht.

---

## 7. Grenzen — was das Skript nicht kann

**Die Reihenfolge muss etwas bedeuten.** Der Monitor simuliert einen Lauf, der
Punkt für Punkt anfällt. Sind die Zeilen in zufälliger Reihenfolge, ist der
Abbruchpunkt eine Zufallszahl. Bei Daten aus einer MD-Trajektorie ist die
Reihenfolge natürlich gegeben.

**Die Grauzone um $\rho = 1$ lässt sich nicht wegrechnen.** $\rho=0{,}99$ und
$\rho=1{,}01$ unterscheiden sich in $N_\text{eff}/n$ um 0,003 — mit endlich
vielen Punkten nicht trennbar. Jede Regel, die $\rho=1{,}05$ erkennt, muss bei
$\rho=0{,}95$ gelegentlich fälschlich feuern. Zwischen etwa 0,95 und 1,2 ist die
Aussage deshalb nicht belastbar; das ist Stetigkeit der Gütefunktion, kein
Umsetzungsfehler.

**Korrelierte Daten machen den Monitor zu selbstsicher.** Die Standardfehler
setzen unabhängige Punkte voraus. Ein realer MD-Strom ist autokorreliert; bei
einer Lag-1-Korrelation von 0,4 wären die Bänder rund 50 % breiter, als das
Skript annimmt. Wenn die Frames dicht aufeinanderfolgen: ausdünnen, oder das
Ergebnis als optimistisch lesen.

**$N_\text{eff}$ misst Ungleichheit, nicht Abdeckung.** Ein Modell kann perfekt
gleichmäßige Gewichte haben und trotzdem eine wichtige Region des
Konfigurationsraums nie besucht haben. Kein Screening kann das ausschließen.

**Das Reweighting korrigiert auf die DFT-Referenz, nicht auf die Realität.**
Fehler des Funktionals bleiben unangetastet.

**Bei `UNKLAR` (Exit 4)** ist eine Voraussetzung verletzt — meist $\hat k \ge
0{,}5$. Dann ist der ausgegebene $N_\text{eff}$-Wert eine Stichprobenzahl ohne
Populationsgrenzwert. Das ist kein Programmfehler, sondern eine Eigenschaft der
Daten: die Gewichtsverteilung hat einen so schweren Schwanz, dass ein einzelner
Frame beliebig dominant werden kann.

Ein Hinweis zu $\hat k$: der Schätzer ist stark verrauscht, sein Standardfehler
fällt nur wie $n^{-1/4}$. Bei wenigen hundert Punkten kann ein einzelner Wert die
0,5-Schwelle nicht sicher entscheiden. Deshalb rechnet das Skript $\hat k$ immer
auf dem **vollen** Satz, nie auf einem Präfix.

---

## 8. Konventionen und Reproduzierbarkeit

$c$ wird mit `ddof=1` gebildet (erwartungstreue Varianz), $\gamma_1$ und
$\gamma_2$ mit `ddof=0` (Plug-in, entspricht `scipy.stats.skew`/`kurtosis` mit
`bias=True`). Das ist die Konvention der Referenzimplementierung. Der Unterschied
zwischen den Konventionen liegt bei $n>100$ unter 0,5 % und damit weit unter dem
Standardfehler von $c$ selbst.

Die Gewichte werden als $w = e^{-\beta(\Delta E - \min\Delta E)}$ gebildet. Der
Abzug macht den größten Exponenten exakt null: Überlauf ist damit ausgeschlossen,
möglich bleibt nur Unterlauf der ohnehin vernachlässigbaren Gewichte. Da
$N_\text{eff}$ gegen einen konstanten Offset in $\Delta E$ invariant ist, ändert
das am Ergebnis nichts.

Der Bootstrap für $\mathrm{SE}(c_\text{max})$ ist über `--seed` reproduzierbar.
Bei $B=200$ streut die geschätzte Bandbreite selbst um etwa 5 %; wer knapp an der
Entscheidungsgrenze liegt, sollte `-B 1000` nehmen.

Warum $\mathrm{SE}(c_\text{max})$ gebootstrappt und $\mathrm{SE}(c)$ analytisch
gerechnet wird: für $c$ trifft die Delta-Methode
$\mathrm{SE}(c) = c\sqrt{(\gamma_2+2)/4k}$ den Bootstrap derselben Stichprobe auf
3 %. Für $c_\text{max}$ versagt die analoge Rechnung — dort steht $f'(c_\text{max})$
im Nenner und geht bei verrauschtem $\hat\gamma_1$ gegen null.

---

## 9. Ein vollständiges Beispiel

```bash
$ python3 kish_screening.py dft.npy mace_l0.npy --steps
==============================================================
  KISH-SCREENING — traegt das Reweighting?
==============================================================
  Punkte n            400
  Temperatur          292.0 K   (beta = 39.742 1/eV)
  Ziel R              0.8

  std(dE)             31.142 meV
  c = beta*std(dE)    1.2376
  Schiefe   gamma1    +0.2481
  Kurtosis  gamma2    +0.1098

  c_max (Gauss)       0.4724
  c_max (schief)      0.5002   <- verwendet
  rho = c/c_max       2.474

  N_eff/n (exakt)     0.3606   < R = 0.8
  khat (Tail-Index)   +0.050   Gate bestanden
  Restglied (2c)^5/5! 0.7743   Reihe unbrauchbar

--------------------------------------------------------------
  SEQUENZIELLER MONITOR
--------------------------------------------------------------
        k        c    c_max   Abstand     Band    Urteil
       50   1.1223   0.4892   +0.6332   0.1491      FAIL

  -> Abbruch bei k = 50 von 400
     88 % der DFT-Punkte waeren gespart worden.

==============================================================
  URTEIL: FAIL
  N_eff/n = 0.3606 < R = 0.8; Monitor feuert bei k = 50
==============================================================

$ echo $?
1
```

Zu lesen als: das Modell verfehlt das Ziel deutlich ($\rho = 2{,}5$), und der
Monitor hätte das schon nach 50 von 400 DFT-Rechnungen gewusst. Die Warnung zum
Restglied ist konsistent — bei $c=1{,}24$ ist die Reihe wertlos, weshalb
$c_\text{max}$ hier nur ein grober Anhalt ist. Das FAIL steht trotzdem, denn es
stützt sich auf den exakten Kish-Wert 0,3606, der ohne jede Reihenentwicklung
auskommt.

---

## 10. Methodik

Der Formalismus ist die auf Modellfehler übertragene Form eines etablierten
Kriteriums: $\beta\sigma_{\Delta U}$ ist in der Freie-Energie-Störungstheorie
seit Zwanzig die Kenngröße für Konvergenz, mit der Faustregel
$\beta\sigma \lesssim 1$. Wu & Kofke (JCP 123, 054103 und 084109, 2005) bauen
darauf ein Bias-Maß über den Phasenraum-Überlapp, allerdings unter Gauß-Annahme —
die $\gamma_1c^3$-Korrektur schließt genau diese Lücke.

Anwendungskontext: Hilpert & Kresse, *Accurate thermophysical properties of water
using machine-learned potentials*, J. Chem. Phys. 164, 194504 (2026).

Der $\hat k$-Schätzer folgt Zhang & Stephens (2009) in der Form, die Vehtari
et al. (JMLR 25, 2024) für PSIS benutzen.
