# Review Runde 3 — getrennt nach Text und Code

Stand: 91 Zellen (68 Markdown, 23 Code), läuft in 98 s durch, 0 Fehler.

---

# Teil 1 — Text und Inhalt

## Falsche oder widersprüchliche Aussagen

### T1. §2.6: „Die Gauß-Version ist konservativ" — widerlegt, aber die Widerlegung wurde gelöscht

> „Bei **Rechtsschiefe** ($\gamma_1>0$) liegt sie **höher** als die Gauß-Schranke […] Die
> Gauß-Version ist also **konservativ** (lehnt im Zweifel eher ab, nie fälschlich an)"

Das stimmt nicht. Der Nulldurchgang der Zonenbreite liegt bei
$\gamma_1^\ast \approx \tfrac{7}{12}\gamma_2 c_\text{max} \approx +0{,}107$, nicht bei 0 —
für $0<\gamma_1<\gamma_1^\ast$ ist die Gauß-Schranke **zu hoch**, also anti-konservativ. Das
war im alten Kapitel 7 gemessen und belegt; mit dessen Streichung ist die Korrektur
verschwunden, die falsche Aussage in §2.6 aber geblieben. **Regression.**

### T2. §2.6-Lesart widerspricht §8.2 direkt

> „Für die **Entscheidung** genügt schon die Gauß-Schranke […] die Korrektur zählt nur bei
> einem Modell nahe der Schwelle"

§8.2 misst: die Gauß-Schranke verwirft ein nachweislich brauchbares Modell in **21 %** der
Sequenzen, die schiefe-korrigierte in 0,6 %. Die Aussage in §2.6 ist das Gegenteil des
Ergebnisses, das zwei Kapitel später steht.

### T3. §6.2: „Bias in die sichere Richtung" — unter der einseitigen Regel falsch herum

> „$\gamma_1,\gamma_2$ sind nach unten verzerrt […] Das ist die **sichere Richtung**:
> […] konservativer, nie fälschlich PASS"

Unter der FAIL-only-Regel ist „konservativer" = **mehr Falsch-FAIL** = die einzige
Fehlerart, die es noch gibt. Der Hinweis darauf stand im alten §8.6 und ist mit der
Umnummerierung verlorengegangen. **Regression.**

### T4. §8.3 nennt Zahlen, die §8.2 nicht mehr liefert

> „Das ist der Grund, warum §8.2 die kausale Variante schlechter abschneiden sieht als die
> Orakel-Variante (**14,9 % gegen 7,1 %** bei $\rho=0{,}9$)"

§8.2 zeigt jetzt **0,6 % gegen 0,6 %** — die beiden sind gleichauf. Der Satz ist in Zahl
*und* Richtung überholt (die Aussage stammt aus der Fassung mit erstem Blick bei k=5).

### T5. §8.5 nennt eine überholte Zahl

„senkt den Fehlalarm bei $\rho=0{,}9$ von 3,6 % auf **0,3 %**" — §8.3 zeigt **0,1 %**.

### T6. §8.3: „von ~400 auf 12 Blicke"

Das Raster hat jetzt **8** Blicke (`checkpoint_grid` liefert 8 für jedes n).

## Verweise, die ins Leere zeigen

### T7. Das k̂-Gate wird fünfmal falsch verortet

§2.3 („prüft §8"), §3-Tabelle („§3.2, §9"), §3.2 („§9"), §5.1 („Details §8"), §5.4 („§8").
Weder §8 noch §9 behandeln k̂ — es steht seit Runde 2 in **§7.2**.

### T8. §8.5 verweist auf §1.5, das gelöscht wurde

„validiert ist die Methodik, nicht die Güte eines konkreten L0-Modells (§1.5)". Kapitel 1
hat nur noch 1.1–1.3. Die Scope-Aussage selbst ist wichtig und sollte irgendwo stehen —
derzeit steht sie nirgends.

### T9. Weitere veraltete Nummern

§2.5-Lesart „$c$ ist robust (§7)" → §6. §2.6 „behandelt §7" → §8.2. §7.3-Einleitung
„woran die Entscheidung in **8.4** hängt" → §7.2. §7.3-Lesart „die zwei Läufe mit
$\hat k>0{,}5$ aus **8.4**" → §7.2, und es sind **drei**, nicht zwei. §7.1 „Folgen dieser
Idealisierung stehen in §8.5" → §9.

## Aussagen ohne Beleg

### T10. §4.2, Zeile „L2-c-01 / n = 400"

W = 0,988, p = 3·10⁻³, Gauß-Fehler −1,4 %. `SHOW_400` enthält nur L0-01, L0-c-01 und
ensemble_L2c — mace-L2-c-01 wird bei n=400 durch keine Zelle geschickt. Die Zeile wird von
nichts erzeugt. *(Steht seit Runde 1 offen.)*

### T11. §3.2: „von verschiedenem Typ, deshalb folgt keine aus der anderen"

Die zwei Gegenbeispiele, die diese Unabhängigkeit belegten, sind beim Kürzen entfallen.
Auch der Satz, dass (ii) **nur die linke Flanke** bindet, fehlt jetzt — damit ist unklar,
warum die Bedingung „exponentiell" heißt.

### T12. §5.2 verweist auf sich selbst

„Aber **(5.2)** bei festem $c$ ist $N_\text{eff}/n$ unbestimmt" — steht *in* §5.2 und meinte
die gelöschte Zweipunkt-Konstruktion. Ebenso §5.4: „bestätigt **5.2**: die Skala genügt
nicht". Zusätzlich: §5 springt von 5.2 auf **5.4**, es gibt kein 5.3.

## Überholte Rahmung

### T13. Titel und §1.3 sind noch zweiseitig

Titelfrage: „ob das Reweighting […] gutartige $N_\text{eff}$ liefern wird" — die Regel kann
nur FAIL feststellen. §1.3: „Deshalb steht das Urteil **PASS / FAIL** oft schon nach wenigen
Punkten fest". Ebenso führt §1.3 $c_\text{max}=\sqrt{-\ln R}$ als *die* Schranke ein, also
genau die, die §8.2 verwirft.

### T14. Titel verspricht Gestrichenes

„Verifikation des Codes gegen **analytische Grenzfälle**" — das war das gestrichene
Gamma-Kapitel. Und „fasst die Analysen `12_screening` und `13_sequential_screening`
zusammen" — `12_screening` war der Ensemble-Pfad, der nicht mehr vorkommt.

### T15. §6.4 ist auf den alten Startpunkt geeicht

„genügen die ±20 % bei $k\approx15$ für ein sicheres Urteil" — die Regel schaut erst ab
$k=n/10$. Und „L0: 1.2 vs. **0.47**" nennt die Gauß-Schranke statt der verwendeten (~0,50).

---

# Teil 2 — Code, Zelle für Zelle

## Look-ahead: kein Befund

Alle 23 Code-Zellen geprüft. **In keiner Entscheidungsgröße steckt Zukunftsinformation.**
Im Einzelnen:

| Zelle | Entscheidungspfad | Wahrheit / Anzeige | Urteil |
|---|---|---|---|
| 67 §7.2 | `monitor_boot(dE)` — intern kausal | `stat_D(dE)`, `kish_ratio_rows`, `khat_rows` auf allen 500 | **sauber** — die Vollstichproben-Größen sind als `c(500)`, `c_max(500)`, `N_eff/500` beschriftet und gehen nicht in die Regel ein |
| 70 §7.3 | `run_stats`, `running_kish`, `psis_khat(dE[:kk])` | — | **kausal** |
| 74 §8.1 | `decide_naive`/`decide_checkpoints` mit `g1_run`, `g2_run` aus `run_stats` | `true_verdict` getrennt | **kausal** |
| 78 §8.2 | `fire_variant` mit `run_stats_2d` | Variante „orakel" nutzt wahre γ — **absichtlich und beschriftet** | **sauber** |
| 81 §8.3 | `stat_D(X)` auf Breite k | — | **kausal** |
| 83 §8.3 | `monitor_boot`, `fire_variant` | Wahrheit aus voller Sequenz | **sauber** |
| 86 §8.4 | `monitor_boot` | — | **kausal** |

## Fehler

### C1. §8.4-Abbildung widerspricht dem eigenen Text **[NEU]**

```python
ax.axvline(2.58, color="k", ls="--", lw=1.6)
ax.text(2.62, 55, "Vorschlag\n$q_{1-\alpha}=2.58$", fontsize=8.5)
```

Der Fließtext derselben Zelle schließt: „**$q_{1-\alpha}=1{,}64$ genügt** […] ein größeres
$q$ kauft nichts mehr". Die Abbildung markiert weiter 2,58 als Vorschlag.

### C2. `R` wird in §2.6 als Schleifenvariable missbraucht **[NEU]**

```python
for R in (0.7, 0.8, 0.9):
```

`R` ist ab §7.1 die globale Effizienzschwelle. Beim Lauf von oben ist die Reihenfolge
zufällig günstig (Zelle 24 vor Zelle 65). Führt man Zelle 24 aber **nach** dem
Werkzeugblock erneut aus — beim interaktiven Arbeiten der Normalfall —, steht global
`R = 0.9`, und alle folgenden Zellen rechnen still mit dem falschen Ziel. Umbenennen in
`R_test`.

### C3. §2.6 benutzt das Bracketing, das der Text daneben als falsch beschreibt **[NEU]**

```python
cmax_exact = lambda R: brentq(lambda c: ..., 1e-4, 3.0)
```

Der Markdown direkt darunter erklärt, warum genau dieses naive Bracket bei $\gamma_2<0$
versagt und dass `uq_mace.screening` es deshalb anders macht. Die Zelle selbst benutzt die
alte Fassung. (Für ensemble_L2c geht es gut, deshalb fällt es nicht auf.) Der Import liegt
allerdings erst in §7.1 — entweder vorziehen oder die Zelle auf ein Gitter umstellen.

### C4. §6.3 rechnet mit dem SE, den §6.1 verwirft **[OFFEN seit Runde 1]**

```python
SEc = c0*np.sqrt((g20+2)/1600); SEg1 = np.sqrt(6/400); SEg2 = np.sqrt(24/400)
```

§6.1 schließt ausdrücklich: „Deshalb schätzen wir SE(γ₁) und SE(γ₂) **ausschließlich per
Bootstrap**". Zwei Zellen später steht die Normaltheorie-Formel, deren Untauglichkeit die
Zelle davor (§6.2) numerisch zeigt (Bootstrap 0,37 gegen Formel 0,24 für γ₂). Zusätzlich
ist `SEc` berechnet und wird nie benutzt.

### C5. §7.3 zeichnet eine Schranke aus einem einzelnen Lauf **[NEU]**

```python
c_fin, g1_fin, g2_fin = c[-1], g1[-1], g2[-1]     # ausserhalb der rep-Schleife
a1.axhline(cmax_skew(R, g1_fin, g2_fin), ...)
```

`c`, `g1`, `g2` stammen aus der **zehnten** Wiederholung — die eingezeichnete Linie ist die
Schranke *eines* zufälligen Laufs, beschriftet aber als wäre sie die Schranke des Modells.
Median über die zehn wäre ehrlicher.

### C6. Kleinigkeiten

- **§4.1**: `SHOW = SHOW_400` — tote Variable.
- **§5.1**: `psis_khat` wird erneut importiert (steht schon im Setup).
- **§8.4**: benutzt `norm.cdf`, aber `norm` wird in §8.1 importiert — stille
  Zell-übergreifende Abhängigkeit.
- **§7.3**: Kommentar „`(§8)`" bei `KH_MIN` — §8 behandelt k̂ nicht.
- **§7.2**: doppelter Kommentar in einer Zeile (`# wenige Sequenzen -> B hoch  # der Monitor`).
- **§8.1**: `ks_all` wird zweimal zugewiesen (einmal vor, einmal in der Schleife).
- **Offset der Gewichte** ist uneinheitlich: mal `- dE.min()` (§7.2), mal `- dE.mean()`
  (§8.1, §8.2, §8.3). Wegen Skaleninvarianz gleichwertig, aber `min` ist numerisch die
  sichere Wahl (alle Gewichte ≤ 1).
- **Sequenzziehung** ist uneinheitlich: §8.3 erzeugt den RNG **vor** der Modell-Schleife
  (verschiedene Ziehungen je Modell), §8.4 **in** der Schleife mit festem Seed (identische
  Ziehungen je Modell). Beides vertretbar, aber es sollte eine Regel geben.

---

# Priorität

1. **T1, T2, T3** — inhaltlich falsche Aussagen, alle drei Regressionen aus dem Löschen
   von Kapiteln. Die Korrekturen existierten schon einmal.
2. **C1, C2** — die Abbildung widerspricht dem Text; `R` als Schleifenvariable ist eine
   stille Falle beim interaktiven Arbeiten.
3. **T4, T5, T6** — veraltete Zahlen.
4. **T7, T8, T9** — Verweise umbiegen (mechanisch).
5. **C3, C4, C5** — Code an die eigene Argumentation angleichen.
6. **T10, T11, T12** — unbelegte Zeile streichen oder erzeugen; §5-Nummerierung schließen.
7. **T13, T14, T15** — Titel und Kapitel 1/6 auf den aktuellen Stand bringen.
8. **C6** — Kosmetik.
