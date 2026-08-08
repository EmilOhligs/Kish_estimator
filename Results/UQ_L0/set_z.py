"""Arbeitspunkt z im Notebook umstellen (§7.8).

    python set_z.py 2.58

Aendert Z im Werkzeugblock (§7.1) und repariert die Stellen, an denen 1.64 noch
festverdrahtet ist und deshalb NICHT mitziehen wuerde. Danach:

    Kernel -> Restart & Run All      (~95 s)

Bewusst NICHT angefasst:
  * z_naive = 1.64 in §7.3  -- dort ist das nominale Niveau der Punkt der Uebung
    (naive Regel gegen Bonferroni-Checkpoints), nicht der Arbeitspunkt.
  * Z_GRID und die Referenzlinie in §7.8 -- das ist der Sweep ueber z selbst.

Was sich durch ein anderes z NICHT aendert: die Schranke c_max (haengt nur an
gamma1, gamma2, R), rho, das wahre N_eff, und alle Diagnosen in §7.6.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

NB = Path(__file__).with_name("screening_methode.ipynb")

# (Suchtext, Ersatztext-Vorlage mit {z}, wie oft erwartet, Beschreibung)
PATCHES = [
    ("Z       = 1.64                     # einseitig nominal 95 %",
     "Z       = {z}                     # Arbeitspunkt, siehe §7.8",
     1, "Werkzeugblock: der eigentliche Knopf"),

    ('print(f"{perms} Bootstrap-Sequenzen je Modell, z=1.64, k>={K_FLOOR}\\n")',
     'print(f"{perms} Bootstrap-Sequenzen je Modell, z={{Z}}, k>={{K_FLOOR}}\\n")',
     1, "§7.2: Kopfzeile beschriftete z fest"),

    ("def fire_variant(D, variant, g1_true, g2_true, z=1.64):",
     "def fire_variant(D, variant, g1_true, g2_true, z=Z):",
     1, "§7.5: fire_variant ignorierte Z (echter Bug)"),

    ('print("4000 Sequenzen a 400 Punkte, z=1.64, k>=5\\n")',
     'print(f"4000 Sequenzen a 400 Punkte, z={{Z}}, k>=5\\n")',
     1, "§7.5: Kopfzeile beschriftete z fest"),
]


def main(z_new: float) -> None:
    if not NB.exists():
        sys.exit(f"Notebook nicht gefunden: {NB}")

    backup = NB.with_name(f"{NB.stem}_vor_z{z_new}.ipynb")
    shutil.copy2(NB, backup)
    print(f"Backup: {backup.name}\n")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    zs = f"{z_new:.2f}"

    total = 0
    for needle, repl, expect, why in PATCHES:
        new = repl.replace("{z}", zs) if "{z}" in repl and "{{" not in repl else repl
        new = new.replace("{{", "{").replace("}}", "}")
        hits = 0
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if needle in src:
                src = src.replace(needle, new)
                lines = src.split("\n")
                cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
                cell["outputs"] = []
                cell["execution_count"] = None
                hits += 1
        flag = "ok" if hits == expect else f"ERWARTET {expect}, GEFUNDEN {hits}"
        print(f"  [{flag:>22}] {why}")
        total += hits

    if total == 0:
        sys.exit("\nNichts geaendert -- steht z vielleicht schon auf einem anderen Wert?")

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nZ = {zs} gesetzt, {total} Stellen angepasst, Outputs der "
          f"betroffenen Zellen geleert.")
    print("\nJetzt: Kernel -> Restart & Run All.")
    print("Danach von Hand nachziehen (die Zahlen stehen in Markdown, nicht im Code):")
    print("  §7.2 Lesart   -- Fehlraten")
    print("  §7.4 Lesart   -- k_med, spaetester Abbruch, Ersparnis")
    print("  §7.5 Ergebnis -- Sweep-Tabelle der drei Varianten")
    print("  §7.6 Ergebnis -- Vergleichstabelle analytisch vs. Bootstrap")
    print("  §7.7 Konsequenz -- Regime-Uebersicht und alle Prozentwerte")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Aufruf: python set_z.py 2.58")
    main(float(sys.argv[1]))
