"""Namenskonflikt aufloesen: Quantil z -> q_{1-alpha}.

    python rename_z_to_q.py            # Probelauf, aendert nichts
    python rename_z_to_q.py --apply    # schreibt

Problem
-------
'z' ist im Notebook doppelt belegt:

  §2.2 ff.  z_i = (dE_i - mu)/sigma   -- Zufallsvariable, eine pro Frame,
                                         Verteilung nachweislich NICHT normal (§4)
  §7        z = 1.64 / 2.58           -- feste Konstante, Quantil der
                                         STANDARDNORMALVERTEILUNG

Beide Konventionen sind fuer sich Standard ('z-Score', 'z = 1.96'), zusammen in
einem Dokument aber irrefuehrend -- zumal §4 gerade zeigt, dass dE nicht normal
ist. Umbenannt wird deshalb das Quantil, nicht die standardisierte Variable:
Letztere steckt in (2.2), (2.3), der gesamten CGF-Herleitung und in §5.2.

Sicherheitsnetz
---------------
Das Skript arbeitet AUSSCHLIESSLICH in den Zellen zwischen <a id="7"> und
<a id="8">. Die Herleitung in §2 und §5 wird per Konstruktion nicht beruehrt.
Am Ende werden verbliebene freistehende 'z' in Kapitel 7 aufgelistet, damit man
den Rest von Hand pruefen kann -- das Skript behauptet nicht, alles zu finden.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

NB = Path(__file__).with_name("screening_methode.ipynb")
NEW = r"q_{1-\alpha}"

# ---------------------------------------------------------------------------
# Regeln. Reihenfolge ist wichtig: spezifisch vor generisch.
# ---------------------------------------------------------------------------
CODE_RULES: list[tuple[str, str, str]] = [
    (r"(?m)^Z(\s*)=(\s*)([0-9.]+)", r"Q_ALPHA\1=\2\3", "Definition Z -> Q_ALPHA"),
    (r"\bZ_GRID\b",   "Q_GRID",  "Z_GRID -> Q_GRID"),
    (r"\bZ_REAL\b",   "Q_REAL",  "Z_REAL -> Q_REAL"),
    (r"\bz_naive\b",  "q_naiv",  "z_naive -> q_naiv"),
    (r"\bz_bonf\b",   "q_bonf",  "z_bonf -> q_bonf"),
    (r"\bz\b",        "q",       "Bezeichner z -> q (inkl. Parameter, Schleifen)"),
    (r"\bZ\b",        "Q_ALPHA", "verbliebene Verwendungen von Z"),
]

MD_RULES: list[tuple[str, str, str]] = [
    (r"\$z\$",                       f"${NEW}$",              "$z$"),
    (r"\$z=",                        f"${NEW}=",              "$z=..."),
    (r"\$z\\ge",                     f"${NEW}\\\\ge",         r"$z\ge..."),
    (r"\$z\\lesssim",                f"${NEW}\\\\lesssim",    r"$z\lesssim..."),
    (r"z\\cdot\\mathrm\{SE\}",       f"{NEW}\\\\cdot\\\\mathrm{{SE}}", r"z\cdot\mathrm{SE}"),
    (r"z\\,\\mathrm\{SE\}",          f"{NEW}\\\\,\\\\mathrm{{SE}}",    r"z\,\mathrm{SE}"),
    (r"\bz-Band\b",                  "Vertrauensband",        "z-Band"),
    (r"\$z\\cdot",                   f"${NEW}\\\\cdot",       r"$z\cdot..."),
    (r"(?<![\w$\\])z = 1\{,\}64",    f"{NEW} = 1{{,}}64",     "z = 1,64 (Fliesstext)"),
    (r"(?<![\w$\\])z=1\{,\}64",      f"{NEW}=1{{,}}64",       "z=1,64 (Fliesstext)"),
    (r"(?<![\w$\\])z=2\{,\}58",      f"{NEW}=2{{,}}58",       "z=2,58 (Fliesstext)"),
]

# Freistehende z, die nach dem Lauf noch in Kapitel 7 stehen -> Restbericht.
# 'z. B.' und Wortbestandteile werden dabei ignoriert.
RESIDUE = re.compile(r"(?<![\w$\\])z(?![\w.])")


def chapter7_range(cells) -> tuple[int, int]:
    start = end = None
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if start is None and '<a id="7">' in src:
            start = i
        elif start is not None and '<a id="8">' in src:
            end = i
            break
    if start is None:
        sys.exit('Kapitel-7-Anker <a id="7"> nicht gefunden.')
    return start, (end if end is not None else len(cells))


def main(apply: bool) -> None:
    if not NB.exists():
        sys.exit(f"Notebook nicht gefunden: {NB}")
    nb = json.loads(NB.read_text(encoding="utf-8"))
    lo, hi = chapter7_range(nb["cells"])
    print(f"Kapitel 7: Zellen {lo} bis {hi - 1}  ({hi - lo} Zellen)")
    print(f"Modus: {'SCHREIBEN' if apply else 'Probelauf (--apply zum Schreiben)'}\n")

    counts: dict[str, int] = {}
    touched_code: list[int] = []

    for i in range(lo, hi):
        cell = nb["cells"][i]
        src = "".join(cell["source"])
        before = src
        rules = CODE_RULES if cell["cell_type"] == "code" else MD_RULES
        for pat, repl, label in rules:
            src, n = re.subn(pat, repl, src)
            if n:
                counts[label] = counts.get(label, 0) + n
        if src != before:
            lines = src.split("\n")
            cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
                touched_code.append(i)

    print("Ersetzungen:")
    for label, n in counts.items():
        print(f"  {n:>4}x  {label}")
    if not counts:
        print("  keine -- schon umbenannt?")

    print(f"\nCode-Zellen mit geleertem Output: {touched_code}")

    print("\nRest zur Handpruefung (freistehendes 'z' in Kapitel 7):")
    found = False
    for i in range(lo, hi):
        src = "".join(nb["cells"][i]["source"])
        for m in RESIDUE.finditer(src):
            ctx = src[max(0, m.start() - 55):m.end() + 45].replace("\n", " ")
            print(f"  [{i} {nb['cells'][i]['cell_type'][:4]}] ...{ctx}...")
            found = True
    if not found:
        print("  keiner")

    if apply:
        backup = NB.with_name(f"{NB.stem}_vor_umbenennung.ipynb")
        shutil.copy2(NB, backup)
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nBackup: {backup.name}")
        print("Geschrieben. Jetzt: Kernel -> Restart & Run All (~95 s),")
        print("damit die Ausgabetexte mit 'z=' nicht mehr auf den alten Namen zeigen.")
    else:
        print("\nNichts geschrieben. Mit --apply erneut aufrufen.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
