#!/usr/bin/env python3
"""Invarianten-Test fuer screening_methode.ipynb.

    python check_notebook.py [--verbose]

Sieben Pruefungen, jede aus einem Fehler entstanden, der in diesem Projekt real
aufgetreten ist. Nach JEDER Aenderung laufen lassen, nicht erst am Ende.
"""
from __future__ import annotations
import ast, builtins, json, re, sys
from pathlib import Path

NB = Path(__file__).with_name("screening_methode.ipynb")
VERBOSE = "--verbose" in sys.argv

# Zahlen, die nominale Niveaus / Literatur / Definitionen sind und deshalb in
# keiner Ausgabe stehen muessen.
WHITELIST = {
    "5,1", "10,0", "2,3", "0,5", "0,1", "5,0", "95,0", "99,0",   # nominale q-Niveaus
    "1,2",                                                        # Dreierregel-Schranke
    "12,0",                                                       # R=0.8 <-> 12 % breiter
}

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
SRC = lambda c: "".join(c["source"])
CODE = [(i, SRC(c)) for i, c in enumerate(cells) if c["cell_type"] == "code"]
MD   = [(i, SRC(c)) for i, c in enumerate(cells) if c["cell_type"] == "markdown"]

def outputs_text(c):
    t = []
    for o in c.get("outputs", []):
        t.append("".join(o.get("text") or []))
        d = o.get("data", {}) or {}
        t.append("".join(d.get("text/plain") or []))
    return "".join(t)

ALL_OUT = "".join(outputs_text(c) for c in cells if c["cell_type"] == "code")
fails: list[str] = []
def report(tag, ok, msgs):
    mark = "OK  " if ok else "FEHL"
    print(f"[{mark}] {tag}" + ("" if ok else f"  ({len(msgs)})"))
    if not ok:
        for m in msgs[:12]:
            print(f"        {m}")
        if len(msgs) > 12:
            print(f"        ... und {len(msgs)-12} weitere")
        fails.append(tag)

# --------------------------------------------------------------- I1 Namen
def scope_names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not node:
            out.add(n.name)
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): out.add(n.id)
        elif isinstance(n, ast.arg): out.add(n.arg)
        elif isinstance(n, ast.Import):
            for a in n.names: out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names: out.add(a.asname or a.name)
        elif isinstance(n, ast.ExceptHandler) and n.name: out.add(n.name)
        stack.extend(ast.iter_child_nodes(n))
    return out

def free_names(node, bound):
    missing, local = [], scope_names(node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs: local.add(a.arg)
        if node.args.vararg: local.add(node.args.vararg.arg)
        if node.args.kwarg: local.add(node.args.kwarg.arg)
    here = bound | local
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            missing += free_names(n, here); continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in here and n.id not in dir(builtins):
                missing.append((n.id, n.lineno))
        stack.extend(ast.iter_child_nodes(n))
    return missing

known, msgs = set(), []
for i, s in CODE:
    try: tree = ast.parse(s)
    except SyntaxError as e:
        msgs.append(f"Zelle {i}: SyntaxError {e}"); continue
    seen = set()
    for name, ln in free_names(tree, known):
        if name not in seen:
            seen.add(name); msgs.append(f"Zelle {i}: '{name}' nicht aufloesbar (Zeile {ln})")
    known |= scope_names(tree)
report("I1  Namensaufloesung (Def vor Use)", not msgs, msgs)

# --------------------------------------------------------- I2 Anker/Verweise
anchors, headings, subsecs = {}, {}, set()
for i, s in MD:
    for m in re.finditer(r'<a id="([\w.]+)"></a>', s): anchors[m.group(1)] = i
    for m in re.finditer(r'(?m)^## (\d+)\.', s): headings[m.group(1)] = i
    for m in re.finditer(r'(?m)^### (\d+\.\d+)', s): subsecs.add(m.group(1))
msgs = []
for num, i in headings.items():
    if num not in anchors: msgs.append(f"Kapitel {num} (Zelle {i}) ohne Anker <a id=\"{num}\">")
for num, i in anchors.items():
    if num.isdigit() and num not in headings:
        msgs.append(f"Anker id=\"{num}\" (Zelle {i}) ohne passende Ueberschrift '## {num}.'")
for i, s in MD:
    for m in re.finditer(r"§(\d+)\.(\d+)", s):
        if f"{m.group(1)}.{m.group(2)}" not in subsecs:
            msgs.append(f"Zelle {i}: Verweis §{m.group(1)}.{m.group(2)} -> kein solcher Abschnitt")
    for m in re.finditer(r"§(\d+)(?!\.\d)", s):
        if m.group(1) not in headings:
            msgs.append(f"Zelle {i}: Verweis §{m.group(1)} -> kein solches Kapitel")
    for m in re.finditer(r"\]\(#(\w+)\)", s):
        if m.group(1) not in anchors:
            msgs.append(f"Zelle {i}: Link (#{m.group(1)}) -> kein solcher Anker")
report("I2  Anker, Ueberschriften, Verweise", not msgs, msgs)

# ------------------------------------------------------- I3 Steuerzeichen
msgs = [f"Zelle {i}: {m.group()!r} an Position {m.start()}"
        for i, c in enumerate(cells)
        for m in re.finditer(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", SRC(c))]
report("I3  keine Steuerzeichen", not msgs, msgs)

# --------------------------------------------------- I4 Zahlen-Provenienz
out_nums = {round(float(x.replace(",", ".")), 3)
            for x in re.findall(r"\d+\.\d+", ALL_OUT)}
out_nums |= {round(float(x), 1) for x in re.findall(r"\d+\.\d", ALL_OUT)}
msgs = []
for i, s in MD:
    for m in re.finditer(r"(\d+),(\d+)\s*%", s):
        lit = f"{m.group(1)},{m.group(2)}"
        if lit in WHITELIST: continue
        val = float(f"{m.group(1)}.{m.group(2)}")
        if not any(abs(val - o) < 0.11 for o in out_nums):
            ctx = s[max(0, m.start()-45):m.end()+25].replace("\n", " ")
            msgs.append(f"Zelle {i}: {lit} % ohne Beleg in einer Ausgabe  ...{ctx}...")
report("I4  Prozentzahlen im Text sind belegt", not msgs, msgs)

# ------------------------------------------------- I5 Abschnittsnummerierung
bykap: dict[str, list[int]] = {}
for sec in subsecs:
    a, b = sec.split("."); bykap.setdefault(a, []).append(int(b))
msgs = []
for kap, nums in sorted(bykap.items()):
    nums.sort()
    if nums != list(range(1, len(nums) + 1)):
        msgs.append(f"Kapitel {kap}: Abschnitte {nums} -- Luecke oder Sprung")
report("I5  Abschnittsnummern lueckenlos", not msgs, msgs)

# --------------------------------------------- I6 Schleifenvariable = Konstante
consts = set()
for i, s in CODE:
    try: tree = ast.parse(s)
    except SyntaxError: continue
    for n in ast.iter_child_nodes(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:                     # auch Tupel-Ziele (a, b = ...)
                for nm in ast.walk(t):
                    if isinstance(nm, ast.Name) and (nm.id.isupper() or len(nm.id) <= 2):
                        consts.add(nm.id)
msgs = []
for i, s in CODE:
    try: tree = ast.parse(s)
    except SyntaxError: continue
    for n in ast.iter_child_nodes(tree):      # nur Modulebene; Schleifen in
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name):   # Funktionen sind lokal
            if n.target.id in consts:
                msgs.append(f"Zelle {i}: 'for {n.target.id} in ...' ueberschreibt eine "
                            f"modulweite Konstante")
report("I6  keine Schleifenvariable auf Konstante", not msgs, msgs)

# ------------------------------------------------------------- I7 Ausgaben
msgs = [f"Zelle {i}: Code ohne Ausgabe" for i, c in enumerate(cells)
        if c["cell_type"] == "code" and not c.get("outputs")]
msgs += [f"Zelle {i}: {o['ename']}" for i, c in enumerate(cells)
         if c["cell_type"] == "code" for o in c.get("outputs", [])
         if o.get("output_type") == "error"]
report("I7  alle Code-Zellen ausgefuehrt, fehlerfrei", not msgs, msgs)


# ------------------------------------------------- I8 Phrasen-Register
# Aus Runde 4: eine Aussage korrigieren und ihre Wiederholungen stehenlassen war
# der haeufigste Fehler. Jede Zeile bindet eine Formulierung an eine Bedingung.
PHRASES = [
    # (Muster, Bedingung, Erklaerung)
    (r"sichere Richtung",        r"zweiseitig",
     "nur mit Bezug auf die ZWEIseitige Regel -- einseitig ist 'konservativ' der Fehler"),
    (r"nie f\u00e4lschlich PASS", r"zweiseitig",
     "dito"),
    (r"kontrolliert und konservativ", None,
     "verboten: unter der einseitigen Regel ist konservativ die einzige Fehlerart"),
    (r"garantiert die Existenz", None,
     "verboten: khat ist ein Schaetzer, Existenz ist an endlichem k nicht beweisbar (§3.2)"),
    (r"klares PASS",             None,
     "verboten: die Regel behauptet nie PASS (§7.1)"),
    (r"exakte Kumulantenformel", None,
     "verboten: die Reihe ist abgebrochen, nicht exakt"),
    (r"fasst \*\*ein einziger dimensionsloser Parameter\*\* zusammen", None,
     "verboten: c SKALIERT, es fasst nicht zusammen (Widerspruch zu §5.2)"),
]
msgs = []
for i, s_md in MD:
    for para in s_md.split("\n\n"):
        for pat, need, why in PHRASES:
            if re.search(pat, para):
                if need is None or not re.search(need, para):
                    frag = re.search(pat, para).group()
                    msgs.append(f"Zelle {i}: '{frag}' -- {why}")
report("I8  Phrasen-Register (korrigierte Aussagen)", not msgs, msgs)

# ------------------------------------------------ I9 Markdown-Fettschrift
msgs = []
for i, s_md in MD:
    for para in s_md.split("\n\n"):
        pos = [m.start() for m in re.finditer(r"\*\*", para)]
        if len(pos) % 2:
            msgs.append(f"Zelle {i}: ungerade Zahl von '**' im Absatz -- unbalanciert")
        elif len(pos) >= 4 and "\n" in para[pos[0]:pos[1]]:
            # erster Bold-Span geht ueber Zeilenumbrueche UND es folgen weitere
            # Marker -> Verschachtelung, die Betonung kehrt sich um
            frag = para[pos[0]:pos[0]+55].replace("\n", " ")
            msgs.append(f"Zelle {i}: mehrzeiliger Bold-Span mit weiteren '**' -- "
                        f"Verschachtelung? ...{frag}...")
report("I9  Fettschrift balanciert, nicht verschachtelt", not msgs, msgs)

print()
print(f"{len(cells)} Zellen | {len(CODE)} Code | {len(MD)} Markdown")
print(("ALLE INVARIANTEN GRUEN" if not fails else f"OFFEN: {', '.join(fails)}"))
sys.exit(1 if fails else 0)
