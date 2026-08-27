# 13 — Sequenzielles Screening

**Frage:** Wie wenige DFT-Einzelpunkte genügen, um ein Modell als tauglich oder
untauglich einzustufen — statt den ganzen Testsatz (oder gar 5000 Frames)
durchzurechnen?

**Anlass:** Aus der Projektgruppe kam die Anregung, dass das Einsparen von DFT-Rechnungen speziell bei den
schnellen, ungenauen **L0-Modellen** interessant wäre. Für ein *schlechtes* Modell
muss man nicht 400 Punkte rechnen, um zu sehen, dass das Reweighting nicht trägt —
$c = \beta\,\mathrm{std}(\Delta E)$ konvergiert schnell und liegt weit über der Schwelle.

Skript: `sequential_screening.py` · Ausgaben: `sequential_workflow_R*.png`,
`sequential_khat_R*.png`, `sequential_screening_R*.csv`

> **Wann sich das Sparen lohnt — Kostenlogik.** Entscheidend ist, welcher Teil teuer ist.
>
> | Modell | MD | DFT | teurer Teil | sequenzielles DFT-Sparen |
> |---|---|---|---|---|
> | **L2** (groß, langsam) | hoch | mittel | MD | bringt wenig (zirkulär) |
> | **L0** (klein, schnell) | niedrig | mittel | **DFT** | **lohnt sich** |
>
> Bei **L0** ist die MD billig, also dominieren die DFT-Einzelpunkte die Kosten. Rechnet
> man sie auf den MD-Frames **nacheinander** und sieht früh, dass $c$ weit über der
> Schwelle liegt, bricht man ab und spart den Großteil der DFT-Kampagne — statt eine
> aussichtslose Reweighting-Rechnung zu Ende zu ziehen. Das ist der eingangs genannte
> Anwendungsfall. Die Konvergenz von $c$, $\gamma_1$ bei kleinem $k$ (unten) ist damit
> **direkt die Auszahlung**: sie bestimmt, nach wie wenigen DFT-Punkten das Urteil steht.
>
> Bei **L2** dominiert dagegen die MD; dort spart frühes DFT-Abbrechen wenig. Und die
> Zahl der Reweighting-Frames $n = 5000$ selbst liegt fest (Volumen-Resampling,
> `notebooks/map.md`) — reduziert wird nicht das finale $n$, sondern der DFT-Aufwand für
> die *Entscheidung*, ob sich der Lauf überhaupt lohnt.

---

## Die Idee in einem Satz

Man rechnet die DFT-Punkte **nacheinander**, führt nach jedem die drei
Diagnosegrößen mit ($c$, $\hat k$, $N_\text{eff}$-Prognose) und **entscheidet
laufend**: PASS / FAIL / weiterrechnen. Sobald das Urteil gesichert ist, hört man auf.

---

## Was simuliert wird

Es liegen 400 reale $\Delta E = E_\text{DFT} - E_\text{MACE}$ pro Modell vor. Der
sequenzielle Prozess wird per **Bootstrap** nachgestellt: jede simulierte Rechen-
Reihenfolge entsteht durch Ziehen **mit Zurücklegen** aus den 400 Werten.

**Warum Bootstrap und nicht Permutation.** Permutation zieht *ohne* Zurücklegen; das
Unsicherheitsband kollabiert dann künstlich bei $k = n$ (endliche Population, Faktor
$\sqrt{(n-k)/(n-1)}$). Im echten Lauf zieht man aus einer langen Trajektorie, also aus
einem faktisch unendlichen Pool. Bootstrap modelliert genau das: die Streuung bei
großem $k$ bleibt erhalten und entspricht dem echten Stichprobenfehler

$$\frac{\mathrm{SE}(c)}{c} \approx \sqrt{\frac{\gamma_2 + 2}{4k}}.$$

Über viele Bootstrap-Sequenzen (Default 800) entstehen die 5–95 %-Bänder in allen
Plots.

---

## Die drei Größen und ihre Rollen

| Größe | Rolle | Panel |
|---|---|---|
| $c = \beta\,\mathrm{std}(\Delta E)$ | **Entscheidungs-Gate.** Konvergiert am schnellsten (stabil ab $k\approx14$). | ① |
| $\hat k$ (PSIS-Tail) | **Existenz-Prüfung — nur auf dem PASS-Zweig** (siehe Asymmetrie unten). | ② |
| $N_\text{eff}/n$-Prognose | **Was $c$ verspricht** — geprüft gegen die exakte Kish-Wahrheit. | ③ |

---

## Die Asymmetrie — warum $\hat k$ nur PASS absichert, nicht FAIL

$N_\text{eff}/n$ ist über Cauchy–Schwarz **monoton fallend** in der Tail-Schwere: ein
schwererer Gewichts-Tail erhöht $E[w^2]$ stärker als $E[w]^2$, drückt $N_\text{eff}$ also
**immer nach unten, nie nach oben**. Nicht-Existenz ($E[w^2]=\infty$) ist der Grenzfall
$N_\text{eff}/n \to 0$.

Daraus folgt eine **Asymmetrie** der beiden Entscheidungszweige:

| Zweig | gebraucht wird | $\hat k$ nötig? |
|---|---|---|
| **FAIL** ($c > c_\text{max}$) | nur eine Obergrenze für $N_\text{eff}$ | **nein** |
| **PASS** ($c < c_\text{max}$) | dass $N_\text{eff}$ auch wirklich **existiert** | **ja** |

Ein unentdeckter schwerer Tail kann nur ein **PASS fälschen** (man zertifiziert 0.9,
während ein ungesehener Extremframe den wahren Wert darunter zieht). Ein **FAIL** kann er
nicht fälschen — dort macht der Tail die Sache nur schlimmer, also die Entscheidung
sicherer. $\hat k$ ist ein Wächter gegen **falsches Akzeptieren**, nicht gegen falsches
Ablehnen.

**Zweiter Grund für die Robustheit von FAIL:** $c = \beta\,\mathrm{std}(\Delta E)$ hängt an
der Streuung von $\Delta E$, **nicht** an der von $w$. Der Rumpf von $\Delta E$ ist
gutartig, selbst wenn $w = e^{-\beta\Delta E}$ einen schweren Tail hat. Numerisch: eine
Verteilung, bei der $N_\text{eff}$ nicht existiert (driftet 0.29 → 0.036 mit wachsendem
$n$), liefert trotzdem ein sauber konvergierendes $c$ — weil $c$ nur $\mathrm{Var}(\Delta E)$
braucht (endlich), $N_\text{eff}$ aber $E[w^2]$ (divergent).

> **Konsequenz für das Verfahren:** $\hat k$ ist kein Vorschalt-Gate für alle Modelle,
> sondern eine **Nachprüfung nur für PASS-Kandidaten**. Seine langsame Konvergenz
> ($n^{-1/4}$) ist damit weit weniger schlimm — man braucht es nur dort, wo $c$ schon
> grünes Licht gibt und ohnehin viele Punkte vorliegen. Auf dem FAIL-Zweig wird es nicht
> ausgewertet.

**Warum $\hat k$ auf dem Testsatz überhaupt nur eine Nachprüfung ist.** Die Existenz von
$N_\text{eff}$ ($E[w^2]<\infty$) ist äquivalent dazu, dass $w$ beschränkt ist, also dass
$\Delta E$ nach unten beschränkt ist — dass es **keine Konfiguration mit beliebig großem
Fehler** gibt. Das ist wörtlich die Abdeckungsfrage A3 (kein Loch in der Fläche). Der
Testsatz enthält solche seltenen Ereignisse per Definition kaum; die Beschränktheit der
400 beobachteten Gewichte ist **kein** Beweis für die Beschränktheit der Verteilung (das
Sample-Maximum ist immer endlich). Deshalb gehört die eigentliche Tail-Überwachung an die
**Produktionstrajektorie** — als laufendes $\hat k$ oder, schneller, als CV-Drift /
Max-Einfluss über die 5000. Siehe `map.md` (A3) und die Diagnosen unten.

**Schnellere Existenz-Diagnosen (ohne GPD-Fit).** Als Ergänzung oder Ersatz für $\hat k$
auf dem PASS-Zweig, alle mit $n^{-1/2}$ statt $n^{-1/4}$:

| Diagnose | Signal | L2c | L0-c-01 |
|---|---|---|---|
| Einfluss des größten Frames $w_\text{max}^2/\sum w^2$ | Anteil an der Degeneration | 1.0 % | 13.8 % |
| $\Delta N_\text{eff}$ beim Entfernen von $w_\text{max}$ | Robustheit | 0.2 % | 9.4 % |
| CV-Drift (2. Hälfte vs. 1. Hälfte) | wächst bei divergenter Varianz | −1.6 % | +8.8 % |

---

## Die Entscheidungsregel

Effizienzkriterium: das Reweighting ist brauchbar, wenn $N_\text{eff}/n \ge R$. Unter
der Gauß-Näherung ist das äquivalent zu einer harten Schranke an $c$:

$$\frac{N_\text{eff}}{n} \ge R \quad\Longleftrightarrow\quad c \le c_\text{max} = \sqrt{-\ln R}$$

Für $R = 0.8$ ist $c_\text{max} = 0.472$. Mit dem einseitigen Stichprobenfehler
(z-Band, Default $z = 1.64$, also 95 %):

$$
\begin{aligned}
c(k) + z\cdot\mathrm{SE}(c) < c_\text{max} &\;\Rightarrow\; \textbf{PASS gesichert}\\
c(k) - z\cdot\mathrm{SE}(c) > c_\text{max} &\;\Rightarrow\; \textbf{FAIL gesichert}\\
\text{sonst} &\;\Rightarrow\; \textbf{weiterrechnen}
\end{aligned}
$$

mit $\mathrm{SE}(c) = c\sqrt{(\gamma_2+2)/4k}$. Das Verfahren stoppt bei der ersten
gesicherten Entscheidung.

---

## Die vier Panels von `sequential_workflow_*.png`

**① Entscheidungs-Gate — $c(k)$ gegen $c_\text{max}$.**
Laufendes $c$ mit 5–95 %-Band. Liegt das Band komplett über $c_\text{max}$ → FAIL,
komplett darunter → PASS. Die L0-Bänder sitzen bei $c\approx1.1$–1.2, weit über der
Schwelle 0.472; die L2-Bänder bei 0.33, klar darunter.

**② Existenz-Prüfung — $\hat k(k)$ gegen 0.5 / 0.25.**
Laufendes $\hat k$ (eigene Bootstrap-Bänder auf einem $k$-Gitter). 0.5 ist die
Existenzbedingung für $N_\text{eff}$, 0.25 die für Fehlerbalken. Diese Prüfung ist nur
für **PASS-Kandidaten** relevant (siehe Asymmetrie oben) — auf dem FAIL-Zweig geht $\hat k$
in keine Entscheidung ein.

**③ Prognose-Validierung — sagt $c$ das endgültige $N_\text{eff}$ voraus?**
Durchgezogen: die $N_\text{eff}/n$-Prognose aus der Kumulantenformel
$\exp(-c^2+\gamma_1 c^3 - \tfrac{7}{12}\gamma_2 c^4)$, laufend. Gepunktet in gleicher
Farbe: der **exakte Kish-Wert** aus allen 400 Punkten. Treffen sich Linie und Punkt-
linie, sagt $c$ das $N_\text{eff}$ korrekt voraus.

> **Lesehilfe zu den breiten Bändern bei L0.** Bei $c>1$ ist das Band riesig — nicht
> als Darstellungsfehler, sondern weil die Formel dort instabil wird: die Terme bis
> $c^4$ hebeln kleine $c$-Schwankungen hoch, und $\gamma_1,\gamma_2$ sind bei kleinem
> $k$ verrauscht. Die **Breite ist selbst diagnostisch**: bei einem guten Modell ist
> die Abbildung $c\to N_\text{eff}$ genau *und* präzise, bei einem schlechten verzerrt
> *und* instabil. Beides ist A2 (Konvergenz der Kumulantenreihe), siehe unten.

**④ Wie früh steht das Urteil fest?**
Verteilungsfunktion des Stopp-Zeitpunkts über alle Bootstrap-Sequenzen. Wo die Kurve
0.95 erreicht, ist in 95 % der Läufe entschieden.

---

## `sequential_khat_*.png` — das Existenz-Gate im Detail

**(a) $\hat k$-Konvergenz.** $\hat k$ gegen die Stichprobengröße mit Band. Das Band
schrumpft langsam ($\propto k^{-1/4}$) — $\hat k$ ist die am schlechtesten bestimmte
Größe. Entscheidend ist nur, ob es die 0.5 sicher unterschreitet.

**(b) Threshold-Stabilität bei $k=400$.** $\hat k$ als Funktion des Tail-Anteils
$M/n$, der in den GPD-Fit eingeht. Ein **Plateau** heißt: das GPD-Modell trägt, die
Wahl der Schwelle ist unkritisch. Steigt die Kurve dagegen mit $M$ an, ist der Fit
fragil — bei `mace-L0-c-01` klettert sie in Richtung 0.35, das Existenz-Gate steht
dort am wackeligsten.

---

## Ergebnisse (Testsatz, 292 K, R = 0.8, c_max = 0.472)

| Modell | $c$(400) | $N_\text{eff}/n$ | $\hat k$ | Urteil | $k_{95}$ | DFT gespart | Fehl |
|---|---|---|---|---|---|---|---|
| mace-L0-01 | 1.238 | 0.361 | 0.050 | FAIL | 10 | 98 % | 0.5 % |
| mace-L0-c-01 | 1.079 | 0.382 | 0.416 | FAIL | 13 | 98 % | 1.1 % |
| mace-L2-c-01 | 0.335 | 0.907 | −0.086 | PASS | 29 | 98 % | 0.0 % |
| ensemble_L2c | 0.327 | 0.912 | 0.068 | PASS | 25 | 98 % | 0.0 % |

**$c$ sagt $N_\text{eff}$ voraus — aber nur für gute Modelle:**

| Modell | Prognose (aus $c$) | Kish (exakt) | Abweichung |
|---|---|---|---|
| ensemble_L2c | 0.912 | 0.912 | −0.03 % |
| mace-L2-c-01 | 0.907 | 0.907 | −0.01 % |
| mace-L0-c-01 | 0.358 | 0.382 | −6.3 % |
| mace-L0-01 | 0.298 | 0.361 | −17.4 % |

**Kernergebnisse:**

- Ein **L0-Modell** ist nach **10–13** DFT-Punkten sicher als untauglich erkannt —
  gegenüber 400 also über 95 % gespart, bei ~1 % Fehlentscheidungen.
- Die c-Formel trifft $N_\text{eff}$ für die **guten** Modelle auf < 0.1 %, für die
  **L0**-Modelle nur auf 6–17 %. Das ist A2: die Kumulantenreihe konvergiert bei
  $c>1$ nicht ($(2c)^5/5! = 0.66$ bei $c=1.24$).
- **Für die Entscheidung ist das egal:** das Urteil läuft über $c$ gegen $c_\text{max}$
  (Panel ①, schmales Band), nicht über den $N_\text{eff}$-Zahlenwert (Panel ③). Das
  Verfahren weiß, *dass* ein Modell schlecht ist, nicht genau *wie* schlecht — für ein
  Aussortieren genau richtig.

---

## Warum die Prognose $\gamma_1$ mitrechnen muss (`moment_convergence.py`)

Die reine Gauß-Schranke $c_\text{max}=\sqrt{-\ln R}$ ist bei **Rechtsschiefe**
($\gamma_1>0$) zu streng: der wahre $N_\text{eff}$ liegt *über* dem Gauß-Wert
($N_\text{eff}/n = e^{-c^2+\gamma_1 c^3 - \frac{7}{12}\gamma_2 c^4} > e^{-c^2}$). Nahe
der Schwelle kann das reine Gauß-Kriterium ein Modell also **fälschlich auf FAIL**
setzen, obwohl sein wahres $N_\text{eff}/n \ge R$ ist. Um das auszuschließen, rechnet die
Prognose $\gamma_1,\gamma_2$ mit und nutzt die **schiefe-korrigierte Schranke** aus

$$c^2 - \gamma_1 c^3 + \tfrac{7}{12}\gamma_2 c^4 = -\ln R.$$

Für L2c ($\gamma_1=+0.50$, $\gamma_2=+0.39$) verschiebt das die Schwelle von
$c_\text{max}=0.472$ auf **0.529** — es sind 12 % mehr $c$ erlaubt. Die
**Falsch-FAIL-Zone** ist der Streifen $0.472 < c < 0.529$ (Panel d): dort sagt Gauß FAIL,
die volle Rechnung PASS.

**Konvergenz der beiden Größen** (Bootstrap, gegen die analytische SE):

| $k$ | $c$ (rel. SE) | $\gamma_1$ | SE($\gamma_1$) | $\gamma_1$-Bias |
|---|---|---|---|---|
| 25 | 0.318 (15.8 %) | +0.368 | 0.44 | −0.13 |
| 100 | 0.323 (7.5 %) | +0.420 | 0.26 | −0.08 |
| 200 | 0.326 (5.7 %) | +0.489 | 0.19 | −0.01 |
| 400 | 0.327 (4.0 %) | +0.491 | 0.13 | −0.01 |

- **$c$ konvergiert schnell** (stabil ab $k\approx15$), genau wie
  $\mathrm{SE}(c)/c=\sqrt{(\gamma_2+2)/4k}$ vorhersagt.
- **$\gamma_1$ konvergiert langsamer** ($\sqrt{6/k}$, stabil erst ab $k\approx200$) und ist
  bei kleinem $k$ **nach unten verzerrt** (0.37 statt 0.50 bei $k=25$). Das ist die
  **sichere Richtung**: eine unterschätzte Schiefe schwächt die Korrektur, die Schranke
  bleibt näher an Gauß, also konservativer. Man riskiert höchstens ein überflüssiges
  Weiterrechnen, nie ein falsches PASS.

**Konsequenz für den Workflow:** $\gamma_1$ (und $\gamma_2$) werden in der Prognose
mitgeführt — nicht für die $N_\text{eff}$-Zahl (die kommt exakt aus Kish), sondern um die
**Entscheidungsschranke** in die richtige, weniger konservative Richtung zu verschieben und
so einen Falsch-FAIL nahe der Schwelle auszuschließen. Weit von der Schwelle (die
L0-Modelle bei $c\approx1.2$) ist die Korrektur belanglos.

---

## Grenzen des Tests

- **Der leichte Fall.** Alle vier Modelle liegen weit von $c_\text{max}$. Der harte
  Fall wäre ein Modell mit $c \approx 0.47$, direkt an der Schwelle — dort entschiede
  das Verfahren nicht nach 15 Punkten, und die Fehlentscheidungsrate stiege in Richtung
  der nominellen 5 %. Solch ein Modell ist in den Daten nicht vorhanden; mit einem
  **synthetischen** $\Delta E$ bei $c\approx0.47$ ließe sich die Grenze gezielt testen.
- **Kein Test von Neuheit/Abdeckung (A3).** Die Bootstrap-Sequenzen ziehen aus den
  realen 400 Frames — geprüft wird die Schätzer*konvergenz*, nicht das Verhalten auf
  ungesehenen Strukturen. Ob die 400 (bzw. die MD) den relevanten Raum abdecken, ist
  eine andere Frage (`validate_ensemble_shift.py`, Teil 3, und `map.md` A3).
- **Optional stopping.** Die Stoppregel prüft nach jedem Punkt neu. Die gemessenen
  1–2 % Fehlentscheidungen liegen unter den nominellen 5 %, aber das z-Band ist nicht
  formal für die vielen Blicke korrigiert. Nahe der Schwelle bräuchte es vorab
  festgelegte Prüfpunkte oder always-valid confidence sequences.

---

## Aufruf

```bash
python analyses/13_sequential_screening/sequential_screening.py
python analyses/13_sequential_screening/sequential_screening.py --R 0.7
python analyses/13_sequential_screening/sequential_screening.py --perms 800 --khat-reps 250
```

Wesentliche Parameter: `--R` (Zielwert), `--z` (Konfidenzniveau der Stoppregel),
`--kmax` (Länge der Bootstrap-Sequenzen), `--perms` / `--khat-reps`
(Bootstrap-Wiederholungen).
