# Review Runde 5 — Text und Code getrennt

Stand: 91 Zellen, 102 s, 0 Fehler, `check_notebook.py` alle neun Invarianten grün.

Weniger Befunde als in Runde 4, und die Fehlerklassen aus den Runden 1–3 sind nicht
wiedergekehrt. **Drei der sechs Textbefunde stammen wieder aus meinen eigenen Runde-4-Edits**
— alle drei aus derselben Ursache: eine Zelle geändert, die Nachbarzeile nicht gelesen.

---

# Teil 1 — Text

## R1. §6.4 widerspricht sich in zwei aufeinanderfolgenden Stichpunkten **[eingeschleppt]**

- Stichpunkt 1: „$c$ … und **die einzige, die in die Schranke $c\le c_\text{max}$ eingeht**"
- Stichpunkt 2: „$\gamma_1,\gamma_2$ … **verschiebt aber die Schranke**"

Beides in derselben Zelle, direkt untereinander. Stichpunkt 1 stammt aus der Zeit, als nur
die Gauß-Schranke benutzt wurde — die hängt tatsächlich nur an $R$. Mit
$c_\text{max}^\text{schief}$ gehen $\gamma_1,\gamma_2$ sehr wohl ein, und §8.3 misst genau
das. Ich habe in Runde 4 Stichpunkt 2 umgeschrieben und Stichpunkt 1 nicht gelesen.

Fix: „…und die einzige, die auf der **linken** Seite des Vergleichs steht; die Schranke
selbst hängt an $\gamma_1,\gamma_2$."

## R2. §8.4-Tabelle: „< 1,5 %" und „1,5 %" bedeuten Verschiedenes **[eingeschleppt]**

In derselben Zeile ($q=1{,}64$) steht „**< 1,5 %**" für $\rho=0{,}90$ und „**1,5 %**" für
$\rho=0{,}95$. Das erste ist eine **obere Schranke** (0 von 200 Sequenzen, Dreierregel), das
zweite ein **Punktschätzer** (3 von 200). Zwei Größen, die sich um eine Größenordnung in
ihrer Aussagekraft unterscheiden, sehen identisch aus.

Fix: Zähler statt Prozent bei kleinen Zahlen — „0/200 (< 1,5 %)" gegen „3/200".

## R3. §8.4: die „7,2 %" beziehen sich auf einen Zustand, den das Notebook nicht mehr enthält

> „Eine frühere Fassung dieses Abschnitts empfahl 2,58 — damals lag der erste Blick bei
> $k=5$, **der Fehlalarm bei 7,2 %**"

Diese Zahl ist im Notebook nirgends mehr erzeugbar. Dieselbe Klasse wie der Befund, den ich
in Runde 4 in §8.3 behoben habe — dort habe ich die Formulierung entschärft, hier ist sie
stehengeblieben. Entweder nach `AENDERUNGEN_kap7_kap8.md` verschieben oder ohne Zahl
formulieren.

## R4. Kapitel 6 rahmt $\gamma_1,\gamma_2$ noch als Prognose-Korrektur **[strukturell]**

§6.3 heißt „Fehlerfortpflanzung" und misst den „Beitrag zu $\log(N_\text{eff}/n)$" in
„% auf $N_\text{eff}$". Der Workflow **prognostiziert aber nie ein $N_\text{eff}$** — er
vergleicht $c$ mit einer Schranke. Relevant wäre also, wie die $\gamma$-Unsicherheit
$c_\text{max}$ verschiebt; genau das misst §8.3 und findet dort $\mathrm{SE}(\hat c_\text{max})
\approx \mathrm{SE}(\hat c)$.

Das ist exakt die Kritik, die schon an §5.2 geäußert wurde („hier wird die Notwendigkeit der
beiden $\gamma$ gezeigt, wenn man $N_\text{eff}$ schätzen will — es sollte um die Schranke
gehen"). §5.2 wurde daraufhin angepasst, §6.3 nie. Der Abschnitt ist nicht falsch, er
beantwortet nur eine Frage, die der Workflow nicht stellt.

Vorschlag: einen Absatz ergänzen, der die Brücke schlägt — „für die Entscheidung zählt nicht
dieser Beitrag, sondern die Verschiebung der Schranke; §8.3" — oder §6.3 auf
$\partial c_\text{max}/\partial\gamma$ umstellen.

## R5. §5.2: Satzfragment **[eingeschleppt]**

„…beweisen lässt es sich an endlichem $k$ nicht (§3.2). **Eine Vorbedingung unabhängig von
$c$ und Form, im Workflow mitgeführt (§7.2).**" — kein Verb.

## R6. §6.2: „die Warnung aus 6.1" ohne §

Kleinigkeit, im Rest des Notebooks durchgehend „§6.1".

---

# Teil 2 — Code

## Look-ahead: erneut kein Befund

Alle 23 Zellen, Schwerpunkt auf den beiden in Runde 4 geänderten (24, 60). Beide sind
Verifikations- bzw. Diagnosezellen ohne Entscheidungslogik. `cmax_exact(R, g1, g2)` nimmt
die Formparameter jetzt als Argumente — die stille Bindung an `ensemble_L2c` ist weg.

## C1. §6.3: `SEg2` wird berechnet und gedruckt, aber nicht verwendet

Die Tabelle zeigt nur den $\gamma_1$-Beitrag (`se3 = c**3*SEg1`). Die Überschrift verspricht
„Beitrag der Terme **und ihre Unsicherheit**" — für den $\gamma_2$-Term steht keine.
Entweder ergänzen ($c^4\,\mathrm{SE}(\gamma_2)\cdot\tfrac{7}{12}$) oder die Überschrift
präzisieren.

## C2. §8.4: $N_{OC}=200$ ist für die Erkennungsspalte grenzwertig

Die Auflösung liegt bei $\pm3{,}5$ Prozentpunkten (bei 50 %). Die Unterschiede in der
Erkennungsspalte — 85,5 → 78,5 → 64,5 — sind damit teils nur zwei Standardfehler. Die
Tabelle lädt zu einer Präzision ein, die sie nicht hat. Ein Satz dazu würde genügen; die
Aussage („später schauen schlägt größeres $q$") trägt die Auflösung problemlos, weil der
relevante Unterschied 78,5 gegen 42,0 ist.

Ich habe $N_{OC}$ in Runde 4 von 250 auf 200 gesenkt, um Laufzeitpuffer zu schaffen — das
war eine stille Verschlechterung der Statistik, die nirgends vermerkt ist.

---

# Teil 3 — Der Checker selbst

## I4 hat eine Lücke: Zufallstreffer über Zellgrenzen

I4 akzeptiert eine Prozentzahl im Text, sobald **irgendeine** Ausgabezahl im ganzen Notebook
innerhalb von 0,11 liegt. Die „7,2 %" aus R3 sind auf diesem Weg durchgerutscht — belegt
angeblich durch 7,1 und 7,13, die aus völlig anderen Zellen stammen.

Zwei Verschärfungen, beide billig:

1. **Toleranz auf 0,05** senken (Rundung auf eine Nachkommastelle braucht nicht mehr).
2. **Herkunft ausgeben** statt nur ja/nein — bei jedem Treffer die belegende Zelle nennen.
   Dann sieht man Zufallstreffer beim Lesen, auch wenn der Test grün ist.

Sinnvoll wäre zusätzlich eine **I10-Prüfung auf Zahlen-Nachbarschaft**: eine Zahl im Text
sollte von einer Ausgabe **derselben oder der unmittelbar vorangehenden Code-Zelle** belegt
sein, nicht von irgendeiner. Das wäre streng genug, um R3 zu fangen, und ließe die
Querverweis-Tabellen (§8.3 zitiert §8.2) über eine kurze Ausnahmeliste zu.

---

# Priorität

1. **R1** — direkter Selbstwiderspruch in einer Zelle.
2. **R2** — zwei Bedeutungen, eine Schreibweise.
3. **I4-Verschärfung** — sonst bleiben Befunde der Klasse R3 unsichtbar.
4. **R3, R5, R6** — Textkorrekturen.
5. **R4** — strukturell, größerer Eingriff; lohnt vor dem Bericht.
6. **C1, C2** — Kosmetik und ein fehlender Vorbehalt.
