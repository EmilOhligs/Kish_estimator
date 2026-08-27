# Kish_estimator

[![tests](https://github.com/EmilOhligs/Kish_estimator/actions/workflows/tests.yml/badge.svg)](https://github.com/EmilOhligs/Kish_estimator/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Uncertainty quantification for MACE machine-learned interatomic potentials,
applied to thermodynamic reweighting of water configurations against DFT
reference energies.

---

## The question

A molecular-dynamics run on a cheap MACE potential samples configurations
$R_i$. To get DFT-accurate thermodynamic averages from that run without
paying for DFT everywhere, you reweight:

$$\langle A\rangle_\text{DFT} \approx \frac{\sum_i A_i w_i}{\sum_i w_i},
\qquad w_i = e^{-\beta(E_\text{DFT}(R_i) - E_\text{MACE}(R_i))}.$$

Whether that reweighting is worth anything depends entirely on how unequal
the weights $w_i$ turn out to be — measured by the **Kish effective sample
size** $N_\text{eff}/n$. This repo is built around answering, and predicting
in advance, that one question: does the reweighting carry, and if not, how
early can you tell (before spending the full DFT budget)?

## Repository structure

| Path | What it is |
|---|---|
| [`kish_screening.py`](kish_screening.py) | Standalone, scipy-free CLI tool — the centerpiece. Decides PASS/FAIL and simulates/runs a sequential early-stopping monitor. Full docs: [`README_kish_screening.md`](README_kish_screening.md) |
| [`src/uq_mace/`](src/uq_mace/) | Core library (`uq-mace` package): reweighting, sequential-screening, ensemble-evaluation and calibration code used across the `analyses/` studies |
| [`analyses/`](analyses/) | 13 numbered, early-stage studies used to build up and stress-test the statistics piece by piece — see [`analyses/README.md`](analyses/README.md) for the full map |
| [`Results/UQ_L0/`](Results/UQ_L0/) | The polished write-up: the complete method end-to-end plus the applied L0-model result, in one notebook |
| [`tests/`](tests/) | `pytest` suite — runs in CI without any of the (gitignored) research data, see below |
| [`tools/`](tools/) | Helper scripts, e.g. [`live_screening_sim.sh`](tools/live_screening_sim.sh) to replay `kish_screening.py`'s live-checkpoint mode against any local cache |

### Why `kish_screening.py` duplicates parts of `src/uq_mace/screening.py`

This is deliberate, not accidental drift. `src/uq_mace/screening.py` is a
library import used by the `analyses/` scripts. `kish_screening.py` is a
**single, dependency-light file** (numpy only) meant to be dropped into an
HPC/production MD pipeline on its own — no package install, no scipy. The
core formulas are kept in sync by construction (both are covered by tests,
[`tests/test_screening.py`](tests/test_screening.py) for the library and
[`tests/test_kish_screening_live.py`](tests/test_kish_screening_live.py) for
the CLI) and were cross-checked bit-for-bit during development.

### From exploration to write-up

`analyses/` is where the statistical framework was built up and tested
piece by piece — skewness corrections, the sequential monitor, the
$\hat k$ tail diagnostic, and so on, each in its own numbered folder.
Once the approach was settled, the complete process — formalism,
validation, and the applied result on real L0-model data — was written up
as one self-contained notebook in
[`Results/UQ_L0/screening_methode.ipynb`](Results/UQ_L0/screening_methode.ipynb).
Start there for the whole story in one place; use
[`analyses/README.md`](analyses/README.md) as a map if you want to dig into
a specific sub-question instead.

**A note on language:** this top-level README is in English, but the
detailed research notes are not — the notebook, `analyses/README.md`,
[`README_kish_screening.md`](README_kish_screening.md), and most code
comments/docstrings are written in German, the working language of the
underlying research project.

### Data, models, cache — intentionally not in this repo

`data/`, `models/`, `cache/` and `notebooks/` are gitignored: unpublished
research data (DFT reference configurations, trained MACE checkpoints,
derived prediction caches) and personal working notes. `kish_screening.py`
itself needs none of this — see the Quickstart below, which works from
synthetic data alone. The `analyses/` scripts and `tests/test_data_smoke.py`
/ `tests/test_cache_konsistenz.py` do need it and skip themselves
automatically when it's absent (as in CI).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quickstart

`kish_screening.py` needs nothing but numpy and runs standalone:

```bash
python3 -c "
import numpy as np
rng = np.random.default_rng(0)
e_dft = rng.normal(0, 0.01, 500)
e_ml  = e_dft + rng.normal(0, 0.003, 500)
np.save('/tmp/e_dft_demo.npy', e_dft)
np.save('/tmp/e_ml_demo.npy', e_ml)
"
python3 kish_screening.py /tmp/e_dft_demo.npy /tmp/e_ml_demo.npy -R 0.8 -T 292
```

Full CLI reference, the statistical formalism, and the live-checkpoint mode
for embedding this into a running MD/DFT campaign are documented in
[`README_kish_screening.md`](README_kish_screening.md).

## Tests

```bash
pytest -q
```

CI runs the full suite on every push (badge above); the data-dependent tests
skip themselves when `data/`/`models/`/`cache/` aren't present, exactly as in
a fresh checkout.

## Methodology & citation

The formalism generalizes $\beta\sigma_{\Delta U}$ from free-energy
perturbation theory (Zwanzig; Wu & Kofke, JCP 123, 2005) with a
skewness-corrected cumulant expansion, and uses the Pareto-$\hat k$
diagnostic of Vehtari et al. (JMLR 25, 2024) as an existence gate for
$N_\text{eff}$. Application context:

> Hilpert & Kresse, *Accurate thermophysical properties of water using
> machine-learned potentials*, J. Chem. Phys. 164, 194504 (2026).

## License

[MIT](LICENSE) for the code in this repository. The underlying DFT/MACE
research data referenced by `analyses/` and `tests/` is **not** included
(see `data/`, `models/`, `cache/` above) and is not covered by this license.
