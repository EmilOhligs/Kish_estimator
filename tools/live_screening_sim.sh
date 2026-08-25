#!/usr/bin/env bash
# live_screening_sim.sh -- Live-Checkpoint-Workflow von kish_screening.py an
# einem bereits fertigen Datensatz durchspielen.
#
# Simuliert eine laufende MD/DFT-Kampagne: zerschneidet einen vollstaendigen
# Cache in wachsende Praefixe (so als kaemen die Punkte gerade nach und nach
# rein) und ruft kish_screening.py --live bei jedem Checkpoint erneut auf --
# genau wie es ein echter MD/DFT-Wrapper taete. Die Checkpoints stammen aus
# derselben Funktion (checkpoints_fuer), die auch das Skript selbst benutzt,
# sind hier also nicht willkuerlich gewaehlt.
#
# Nuetzlich, um den --live-Workflow an eigenen, bereits vorliegenden Daten
# auszuprobieren, ohne dass eine echte laufende Simulation noetig ist.
#
# Aufruf:
#   tools/live_screening_sim.sh CACHE.npz [N_PLAN] [R] [T]
#
# CACHE.npz  Einzeldatei-Cache mit 'e_dft' und 'e_mace'/'energies'
#            (Projekt-Cache-Format, siehe kish_screening.py).
# N_PLAN     geplantes Gesamtbudget. Default: alle Punkte in CACHE.npz.
# R          Zielwert N_eff/n. Default: 0.8
# T          Temperatur in Kelvin. Default: 292
#
# Beispiel:
#   tools/live_screening_sim.sh cache/single_mace-L2-c-01_testfull_n5000_clean.npz 4996 0.8 292

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Aufruf: $0 CACHE.npz [N_PLAN] [R] [T]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CACHE="$1"
R="${3:-0.8}"
T="${4:-292}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 - "$CACHE" "$TMP" <<'PYEOF'
import sys
import numpy as np
sys.path.insert(0, ".")
from kish_screening import lade_paar

e_dft, e_ml = lade_paar(sys.argv[1])
np.save(f"{sys.argv[2]}/dft_bisher.npy", e_dft)
np.save(f"{sys.argv[2]}/ml_bisher.npy", e_ml)
PYEOF
N_VERFUEGBAR=$(python3 -c "import numpy as np; print(np.load('$TMP/dft_bisher.npy').size)")
N_PLAN="${2:-$N_VERFUEGBAR}"

echo "CACHE=$CACHE  N_PLAN=$N_PLAN  R=$R  T=$T"
echo

# dieselben Checkpoints, die das Skript selbst fuer n_plan benutzen wuerde
CHECKPOINTS=$(python3 - "$N_PLAN" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from kish_screening import checkpoints_fuer, K_FLOOR, FIRST_FRAC
n_plan = int(sys.argv[1])
ck = checkpoints_fuer(n_plan, K_FLOOR, FIRST_FRAC)
print(" ".join(str(k) for k in ck if k < n_plan))
PYEOF
)

for k in $CHECKPOINTS; do
    python3 - "$k" "$TMP" <<'PYEOF'
import sys
import numpy as np
k, tmp = int(sys.argv[1]), sys.argv[2]
np.save(f"{tmp}/dft_k.npy", np.load(f"{tmp}/dft_bisher.npy")[:k])
np.save(f"{tmp}/ml_k.npy",  np.load(f"{tmp}/ml_bisher.npy")[:k])
PYEOF
    echo "--- k=$k von $N_PLAN ---"
    python3 kish_screening.py "$TMP/dft_k.npy" "$TMP/ml_k.npy" \
        -N "$N_PLAN" --live -T "$T" -R "$R" -q \
        && echo "Exit: 0" || echo "Exit: $?"
done

echo
echo "=== finaler Aufruf bei n = n_plan = $N_PLAN -> volle Zertifizierung ==="
python3 kish_screening.py "$TMP/dft_bisher.npy" "$TMP/ml_bisher.npy" \
    -N "$N_PLAN" --live -T "$T" -R "$R" --steps
echo "Exit-Code: $?"
