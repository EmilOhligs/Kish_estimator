# Review, Runde 2

Stand: 91 Zellen, Kapitel 1–8 ausgearbeitet, §9 (Grenzen/Autokorrelation) und §10
(Zusammenfassung) als Platzhalter. Die alten Platzhalter §9 (Existenz) und §10
(Gamma-Verifikation) sind gestrichen — das deckt sich mit der Empfehlung aus Runde 1.

**NEU** = in dieser Runde gefunden. **OFFEN** = aus Runde 1, noch nicht behoben.

---

## A — Falsche Zahlen im Text

### A1. §7.2 und §8.5 widersprechen sich bei der Ersparnis **[NEU, verifiziert]**

- §7.2: „90 % gesparte DFT-Punkte gegenüber den 500 des Laufs, und **99 % gegenüber den
  5000** eines Produktionslaufs"
- §8.5: „Der Abbruch fällt beim ersten Checkpoint, also bei $n/10$: **90 %** der Punkte
  gespart, **unabhängig von $n$**"

§8.5 hat recht. Unter der $n/10$-Regel liegt der erste Blick bei $n=5000$ bei $k=500$, die
Ersparnis ist also auch dort 90 %, nicht 99 %. Die 99 % würden nur gelten, wenn man
$k_\text{first}=50$ **absolut** beibehielte — was genau der offene Punkt B1 ist. §7.2 ist zu
korrigieren.

### A2. §8.5 nennt eine veraltete Zahl für die Gauß-Schranke **[NEU, verifiziert]**

§8.5: „Die Gauß-Schranke verwirft ein nachweislich brauchbares Modell in **33 %** der
Sequenzen." Die aktuelle Messung in §8.2 ergibt **21,4 %** (bei $\rho=0{,}90$). Die 33 %
stammen aus der Fassung mit $k_\text{first}=5$.

---

## B — Code-Fehler

### B1. Derselbe RNG zieht Sequenzen und Bootstrap **[NEU, verifiziert]**

In §7.2 (Zelle 67) und §8.3 (Zelle 83):

```python
rng = np.random.default_rng(42)
idx = rng.integers(...)              # zieht die Sequenzen
fired, kfire = monitor_boot(dE, rng=rng)   # derselbe Generator fuer den Bootstrap
```

Folge: die Sequenzen des zweiten Modells hängen davon ab, wie viele Zufallszahlen der
Bootstrap des ersten verbraucht hat. Ändert man `B` oder das Checkpoint-Raster, ändern sich
**auch die gezogenen Sequenzen** — Vergleiche zwischen Einstellungen sind dann nicht mehr
auf identischen Daten. Fix: zwei getrennte Generatoren.

*Das betrifft rückwirkend die Vergleiche „k_first = 5 gegen 50", die ich gemessen habe. Die
Richtung des Befunds ist robust (Faktor 10), die Nachkommastellen sind es nicht.*

### B2. `fire_variant` folgt `Q_ALPHA` nicht **[NEU, verifiziert]**

`def fire_variant(D, variant, g1_true, g2_true, q=1.64)` — hart verdrahtet. Ein geänderter
Arbeitspunkt würde in §8.2 stillschweigend ignoriert. (Dieselbe Stelle war schon in
`set_z.py` als Bug markiert; der Umbenennungslauf hat sie zu `q=1.64` gemacht, aber nicht an
`Q_ALPHA` gebunden.)

### B3. Die Ausgabe in §8.2 behauptet „k>=5" **[NEU, verifiziert]**

`print("4000 Sequenzen a 400 Punkte, q=1.64, k>=5")` — der Code benutzt aber
`fail & (k >= K_FLOOR)` mit `K_FLOOR = 50`. Die Kopfzeile ist schlicht falsch.

### B4. Vier verwaiste Kapitelverweise **[NEU, verifiziert]**

Nach dem Streichen der alten Platzhalter zeigen ins Leere:
Inhaltsverzeichnis (zweimal auf `#11`/`#12`), Zelle 75 („§3.2, §11"), Zelle 88 („§3.2,
§11"). Gemeint ist jeweils das heutige §9.

### B5. `cmax_skew` (skalar) und `cmax_skew_vec` fallen unterschiedlich zurück **[OFFEN]**

Bei γ₁ = 1,4: skalar 0,4724 (Gauß, anti-konservativ), vektorisiert 0,7155 (C_VALID). Beißt
nicht, weil die skalare Fassung nur mit wahren γ aufgerufen wird — aber die Verifikation im
Werkzeugblock würde deshalb einen kaputten Vektor-Code nicht auffangen.

### B6. `CHECKPOINTS` fest für n = 500, benutzt auf 400er-Sequenzen **[OFFEN]**

§8.2/§8.3 arbeiten mit 400 Punkten; der erste Blick liegt dort bei 50/400 = **12,5 %**, nicht
bei den behaupteten 10 %.

### B7. `kh_grid` in §7.3 stammt aus der alten Fassung **[NEU]**

`(15, 30, 60, 120, 240, 400, 500)` mit `KH_MIN = 50`-Filter — die ersten beiden Punkte
werden nur blass gezeichnet. Seit der Umstellung wäre das Checkpoint-Raster die natürliche
Wahl.

### B8. Bootstrap-Rauschen der Bandbreite **[OFFEN]**

`SE_boot(D)` streut bei B = 100 und k = 50 um 8,8 %; das effektive q schwankt damit um ±0,14.
B = 400 halbiert das und kostet bei k = 50 fast nichts.

---

## C — Methodische Punkte

### C1. Die $n/10$-Regel koppelt an die falsche Größe **[OFFEN, weiterhin der wichtigste Punkt]**

Die Gründe für den späten ersten Blick sind **absolut in k**: SE(c)/c = √((γ₂+2)/4k), und die
Nichtkonvergenz der Schranke (43 % bei k=5 → 2 % bei k=50). Bei n = 500 fällt n/10 zufällig
mit dem sinnvollen Wert zusammen; bei n = 5000 verlangt die Regel k_first = 500 statt der
nötigen ~50. Das kostet 450 DFT-Punkte je abgebrochenem Modell und ist zugleich die Ursache
des Widerspruchs A1.

### C2. Das k̂-Gate wird versprochen, aber nicht mitgeführt **[OFFEN]**

§3.2 erklärt E[w²] < ∞ zur Annahme A2 mit k̂ als Diagnose. Der Monitor prüft es nirgends,
§7.2 gibt es nicht aus. Nach dem Streichen des alten §9 hat der Verweis auch kein Ziel mehr.

### C3. Der Bootstrap kann nichts über den ungesehenen Rand wissen **[NEU, Einschätzung]**

`SE_boot(D)` schätzt die Streuung *gegeben die empirische Verteilung der k Punkte*. Fehlt der
Tail noch, ist nicht nur ĉ zu klein (§3.2) — auch das **Band** ist zu schmal. Beide Effekte
zeigen in dieselbe, konservative Richtung, aber der zweite steht nirgends. Ein Halbsatz in
§3.2 oder §8.3 genügt.

### C4. `k` bedeutet zweierlei **[OFFEN]**

Stichprobengröße und Pareto-Tail-Index. In §3.2 stehen „$k<1/r$" und „$k$ Punkte" in
benachbarten Absätzen. Vorschlag: Tail-Index als ξ.

### C5. §8.2 wiederholt §7.1 wörtlich **[NEU]**

Der Abschnitt „Warum PASS keine echte Entscheidung ist" steht vollständig in §7.1 und noch
einmal in §8.2. Nach der Kapitelteilung ist die zweite Fassung überflüssig — §8.2 kann direkt
mit der Schranken-Frage einsteigen (~120 Wörter).

---

## D — Die verbliebenen Platzhalter

**§9 „Grenzen: Abdeckung und Nicht-iid (Fokus Autokorrelation)"** — der einzige inhaltlich
offene Punkt des ganzen Notebooks. Alle Sequenzen sind iid gebootstrappt; ein realer MD-Strom
ist korreliert, dann ist SE(c) um √τ größer als angenommen und der Monitor zu selbstsicher.
Bei k_first = 50 ist das gegenüber k = 5 deutlich entschärft, aber ungemessen. Ein AR(1)-Sweep
mit τ-Werten aus der realen Trajektorie beantwortet es in einer Zelle. **Nötig.**

Die Abdeckungshälfte ist erledigt (§3.2) und braucht dort höchstens die eine Messzelle.

**§10 „Zusammenfassung"** — nötig.

---

## E — Reihenfolge

1. **A1, A2** — falsche Zahlen korrigieren (zwei Sätze).
2. **B4** — verwaiste Verweise auf §9 umbiegen, Inhaltsverzeichnis.
3. **B1** — RNG-Ströme trennen. Danach sind alle Vergleichszahlen erst wirklich belastbar.
4. **C2** — k̂ in die §7.2-Tabelle.
5. **B2, B3, B5, B6, B7** — Aufräumen.
6. **C1** — n/10 gegen absoluten Startwert messen. Der einzige Punkt, der das Ergebnis
   verändern kann.
7. **§9** schreiben (AR(1)), dann **§10**.
8. **C5, C4, B8** — Kosmetik und Feinschliff.
