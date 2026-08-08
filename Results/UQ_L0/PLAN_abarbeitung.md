# Arbeitsplan: Befunde aus Review-Runde 3 gesichert abarbeiten

Ziel: die 15 Text- und 6 Code-Befunde beheben, **ohne** neue einzuschleppen. Der Plan ist
weniger eine Liste von Fixes als eine Liste von Absicherungen — die Fixes selbst sind
größtenteils trivial, das Risiko liegt im Vorgehen.

---

## Was in dieser Sitzung schiefgegangen ist (und was daraus folgt)

| Vorfall | Ursache | Gegenmaßnahme im Plan |
|---|---|---|
| `q_{1-\alpha}` wurde in 29 Zellen zu `q_{1-\x07lpha}` | `re.sub` interpretiert `\a` im **Ersetzungstext** | Regel P3: nie `re.sub` mit Backslash im Replacement |
| Zwei Patch-Skripte starben mittendrin | `str.replace` ohne Prüfung, ob das Ziel existiert | Regel P1: jede Ersetzung asserted, All-or-nothing |
| §7.8 landete hinter Kapitel 9–12 | `cells.append()` statt Einfügen an der Abschnittsgrenze | Regel P2: Position immer über Anker suchen |
| §2.6 und §6.2 behaupten seit dem Löschen von Kapitel 7 Falsches | Korrekturen lebten nur in der gelöschten Zelle | Phase 2 + Claim-Register |
| Anker `id="11"` bei Überschrift „## 9." | Umnummerierung ohne Ankerprüfung | Checker-Invariante I2 |
| Zahlen im Text veralteten nach jedem Lauf | händisch gepflegt | Phase 1 vor Phase 3, Checker-Invariante I4 |
| `B=200` sprengte das Zeitbudget | Laufzeit nicht vorab geschätzt | Regel P5 |

---

## Phase 0 — Absicherung bauen (vor jeder Änderung)

**0.1 Backup** `screening_methode_vor_runde3.ipynb`.

**0.2 `check_notebook.py` schreiben** — ein wiederholbarer Invarianten-Test. Ohne ihn ist
jede spätere Änderung wieder blind. Er prüft:

- **I1 Namensauflösung** — AST-basierte Def-vor-Use-Analyse über alle Code-Zellen
  (das Skript aus dieser Sitzung, mit korrektem Scope-Handling).
- **I2 Anker ↔ Überschriften** — jede `## N.` hat `<a id="N">`, jedes `§N` und jedes
  `(#N)` zeigt auf einen existierenden Anker, jedes `§N.M` auf eine existierende
  `### N.M`-Überschrift.
- **I3 Steuerzeichen** — kein `[\x00-\x1f]` außer `\n` im gesamten Notebook.
- **I4 Zahlen-Provenienz** — jede Prozentzahl in Markdown, die wie ein Messwert aussieht
  (`\d+[,.]\d\s*%`), muss in der Ausgabe **irgendeiner** Code-Zelle vorkommen. Toleranz
  für Rundung (0,1 pp). Ausnahmen über eine Whitelist (nominale Niveaus, Literaturwerte).
- **I5 Abschnittsnummerierung** — lückenlos je Kapitel (fängt die 5.2→5.4-Lücke).
- **I6 Keine Globals-Kollision** — Schleifenvariablen in Code-Zellen dürfen nicht auf
  Namen laufen, die später als Konstante gesetzt werden (`R`, `c`, `k`, `q`, `beta`).
- **I7 Laufzeit** — Gesamtlaufzeit unter 110 s (Tool-Limit 120 s).

**0.3 Baseline** — Checker laufen lassen, Ist-Zustand festhalten. Jeder Befund, den er
*jetzt schon* meldet, ist einer der bekannten; alles Neue später ist selbstverschuldet.

**0.4 Claim-Register** `CLAIMS.md` anlegen: für jede tragende Aussage die Zelle, die sie
belegt. Drei Zeilen für die drei Regressionen, plus die Kernzahlen. Zweck: wer künftig
eine Zelle löscht, sieht sofort, welche Behauptung damit unbelegt wird. Genau das ist beim
Löschen von Kapitel 7 passiert.

---

## Phase 1 — Erst alles, was Zahlen ändert

Reihenfolge ist zwingend: Text über Zahlen zu schreiben, bevor die Zahlen final sind,
war der häufigste Fehler dieser Sitzung.

**1.1 C2 — `R` als Schleifenvariable** (§2.6). → `R_test`. Danach I6 prüfen.

**1.2 C3 — naives Bracketing** in §2.6. Zwei Wege:
   (a) `scr`-Import vor Kapitel 2 ziehen und `cmax_skew` benutzen — sauber, aber der
       Werkzeugblock gehört didaktisch zu §7.1;
   (b) die Zelle auf dieselbe Gitter-Logik umstellen, mit Kommentar „identisch zu
       `uq_mace.screening.cmax_skew`".
   **Empfehlung (b)** — §2.6 soll ohne die spätere Bibliothek lesbar bleiben.

**1.3 C4 — §6.3 rechnet mit der verworfenen SE-Formel.** Das ist der einzige Punkt in
Phase 1, der echte Zahlen ändert. Vorgehen:
   - `SEg1`, `SEg2` aus dem **Bootstrap** derselben 400 Punkte ziehen (die Zelle §6.2
     berechnet sie bereits — Wert übernehmen statt neu rechnen, sonst kostet es Laufzeit)
   - `SEc` löschen (unbenutzt)
   - erwartete Verschiebung: 0,4 % → ~0,5 %, 12 % → ~13,5 %. **Vorher einmal separat
     rechnen**, damit klar ist, was herauskommen muss.

**1.4 C1, C5 — Abbildungen.** §8.4: `axvline(2.58)`/„Vorschlag" auf 1,64 umstellen.
§7.3: die Schranke als **Median über die zehn Läufe** zeichnen statt aus dem letzten.

**1.5 Lauf + Ausgaben einfrieren.** Restart & Run All, dann alle Zell-Ausgaben in eine
Textdatei extrahieren. Diese Datei ist ab jetzt die **einzige** Quelle für Zahlen im Text.

---

## Phase 2 — Die drei inhaltlichen Fehler (T1, T2, T3)

Diese zuerst nach Phase 1, weil sie auf Messwerte verweisen sollen.

**2.1 T1 — §2.6, „Gauß ist konservativ".** Nicht nur den Satz ändern, sondern **belegen**:
eine kurze Zelle, die $\gamma_1^\ast$ numerisch findet und gegen die Näherung
$\tfrac{7}{12}\gamma_2 c_\text{max}$ hält (beides ergab +0,1075). Damit ist die Aussage
gegen künftiges Löschen immun — sie steht dann dort, wo sie hingehört, statt in einem
Kapitel, das gestrichen werden kann. In `CLAIMS.md` eintragen.

**2.2 T2 — §2.6-Lesart.** Ersetzen durch einen Vorwärtsverweis: „welche der beiden
Schranken die Regel benutzt, entscheidet §8.2 — gemessen, nicht argumentiert (21 % gegen
0,6 % Fehlalarm)."

**2.3 T3 — §6.2, „Bias in die sichere Richtung".** Der Satz stimmt für die zweiseitige
Regel und wird unter der einseitigen falsch. Also nicht streichen, sondern
konditionalisieren: „…unter einer zweiseitigen Regel. Unter der einseitigen Regel aus §7.1
ist genau das die einzige verbleibende Fehlerart — §8.3 beziffert den Preis."

**2.4 Gegenprobe.** Nach diesen drei Änderungen gezielt nach weiteren Sätzen suchen, die
aus der Zweiseitigkeit stammen: Muster `konservativ`, `nie fälschlich PASS`, `PASS / FAIL`,
`sichere Richtung`.

---

## Phase 3 — Verweise und Zahlen synchronisieren

Rein mechanisch, vollständig vom Checker getrieben.

**3.1** k̂-Verweise (§2.3, §3-Tabelle, §3.2, §5.1, §5.4) auf **§7.2** umbiegen.
**3.2** §8.5 → §1.5: die Scope-Aussage ist inhaltlich wichtig und steht nach der Löschung
nirgends. Sie gehört als zwei Sätze in §7.2 oder §8.5 selbst, nicht als Verweis.
**3.3** Restliche Nummern: §2.5→§6, §2.6→§8.2, §7.3-Einleitung 8.4→7.2, §7.3-Lesart
„zwei Läufe" → **drei**, §7.1 „§8.5" → §9.
**3.4** Veraltete Zahlen T4/T5/T6 aus der eingefrorenen Ausgabedatei nachziehen.
**3.5** Checker I2 + I4 müssen grün sein.

---

## Phase 4 — Unbelegte Aussagen

**4.1 T10** — §4.2-Tabellenzeile „L2-c-01 / n=400". Entweder `SHOW_400` um das Modell
erweitern (dann wird sie erzeugt, kostet aber ein viertes Panel) oder Zeile streichen.
**Empfehlung: erzeugen** — die Zeile trägt das Argument des Abschnitts (L2 ist am
wenigsten gaußisch und hat den kleinsten Fehler).

**4.2 T11** — §3.2: entweder die zwei Gegenbeispiele wieder aufnehmen (~60 Wörter) oder
die Behauptung „von verschiedenem Typ, deshalb unabhängig" abschwächen. Ebenso den Satz
zurückholen, dass (ii) nur die linke Flanke bindet — sonst ist „exponentiell" unmotiviert.

**4.3 T12** — §5: Selbstverweise auf die gelöschte Zweipunkt-Konstruktion entfernen,
Nummerierung 5.2/5.4 schließen (I5).

---

## Phase 5 — Rahmung

**5.1 T13** — Titelfrage und §1.3 auf einseitig umstellen; §1.3 darf $\sqrt{-\ln R}$ als
*Einstieg* behalten, muss aber sagen, dass die verwendete Schranke die schiefe-korrigierte
ist (§2.6, §8.2).
**5.2 T14** — Titel: „Verifikation gegen analytische Grenzfälle" und der Verweis auf
`12_screening` streichen.
**5.3 T15** — §6.4 auf den Startpunkt $n/10$ eichen, „0.47" → die verwendete Schranke.

---

## Phase 6 — Abschluss

**6.1** Restart & Run All, 0 Fehler, Laufzeit < 110 s.
**6.2** Checker vollständig grün.
**6.3** Diff gegen `screening_methode_vor_runde3.ipynb` durchsehen — jede geänderte Zelle
einmal ansehen, nicht nur die Zusammenfassung glauben.
**6.4** `REVIEW_runde3.md` abhaken, offene Punkte (C1 aus Runde 2: die $n/10$-Frage) als
solche stehen lassen.

---

## Patch-Disziplin (gilt in allen Phasen)

- **P1 All-or-nothing.** Jede Ersetzung mit `assert ziel in quelle`. Erst wenn *alle*
  Ersetzungen einer Phase durchgelaufen sind, wird geschrieben. Ein gestorbenes Skript
  darf keinen halben Zustand hinterlassen.
- **P2 Position über Anker.** Zellen nie über absolute Indizes einfügen, immer relativ zu
  `<a id=...>` oder `### N.M`. Indizes verschieben sich bei jeder Änderung.
- **P3 Kein `re.sub` mit Backslash im Replacement.** `\a`, `\b`, `\1` werden interpretiert.
  Entweder `str.replace` oder `re.sub(..., lambda m: ersatz)`.
- **P4 Outputs leeren** bei jeder geänderten Code-Zelle, damit nie Ausgabe und Code
  auseinanderlaufen (der Befund aus der allerersten Runde).
- **P5 Laufzeit schätzen**, bevor `B`, `N_OC` oder Sequenzzahlen steigen.
- **P6 Nach jeder Phase Checker.** Nicht erst am Ende.

---

## Reihenfolge in einem Satz

Absicherung bauen → Zahlen fixieren → einfrieren → Inhalt korrigieren → Verweise
nachziehen → Belege schließen → Rahmung → verifizieren.

Der einzige Punkt mit echtem Entscheidungsbedarf bleibt außen vor: ob der erste Blick bei
$n/10$ oder bei einem absoluten $k$ liegen soll (Runde 2, C1). Der gehört nicht in diese
Aufräumrunde, weil er Zahlen im ganzen Kapitel 8 verschiebt.
