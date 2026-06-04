# child-cyclist-crash-reconstruction

MADYMO multibody simulation pipeline for reconstructing a fatal child cyclist collision and characterising head impact conditions for cycle helmet safety assessment.

**MEng Design Engineering capstone project**  
Imperial College London, 2026  
Supervisor: Dr Mazdak Ghajari, HEAD Lab

---

## Project overview

This repository contains the Python analysis pipeline used in the master's thesis:

> *"Simulation of Child Head Impact Conditions for Cycle Helmet Safety Assessment"*

A MADYMO multibody dynamics model reconstructs a fatal real-world collision between a 12-year-old cyclist and a VW Crafter van on a 20% downhill gradient. A sensitivity study of 58 simulation configurations systematically varies six parameters — bicycle velocity, van velocity, bicycle lean angle, velocity angle, neck flexion and lumbar rotation — to characterise the distribution of head impact conditions. Results are compared against EN 1078 and EN 1080 helmet certification test conditions to assess their representativeness for real-world child cyclist impacts.

---

## Repository structure

```
├── simulation_functions.py   Core functions: CSV parsing, impact metric extraction,
│                             HIC15 calculation, results saving
├── extract_results.py        Batch extraction of peak impact metrics from completed
│                             MADYMO simulation runs (no re-running required)
├── generate_figures.py       All publication-quality figures for the sensitivity
│                             study results (Figures 1–9 in report)
└── README.md
```

---

## Dependencies

Python 3.10+ with the following packages:

```
numpy
pandas
scipy
matplotlib
openpyxl
```

Install with:

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

## Usage

### 1. Run MADYMO simulations

Simulations are run using MADYMO 2021.1 via the `AutorunMADYMO_v2.py` script (not included here — contains lab-specific pipeline code). Each run produces CSV output files in `simResults/<run_number>/`.

### 2. Extract results

Edit the paths and `sheetNames` at the top of `extract_results.py` to match your simulation folder, then run:

```bash
python extract_results.py
```

This reads the CSV output files from each completed run and saves peak impact metrics (HIC15, head acceleration, helmet force, neck force, impact speed and angle, Euler angles) to an Excel file in your Results folder.

### 3. Generate figures

Edit the `filePath_results` and `filePath_figures` paths at the top of `generate_figures.py`, then run:

```bash
python generate_figures.py
```

Figures are saved as 300 dpi PNG files. Each figure block is self-contained — comment out any you do not need.

---

## Sensitivity study parameters

| Parameter | Baseline | Range |
|-----------|----------|-------|
| Bicycle velocity (m/s) | 6.1 | 6.1 – 10.768 |
| Van velocity (m/s) | 2.0 | 0 – 2.5 |
| Bicycle lean angle, R3 (rad) | 0.1 | −0.1 – 0.15 |
| Velocity angle, R3add (rad) | −0.18 | −0.25 – 0 |
| Neck flexion, NeckLow R2 (rad) | 0.25 | 0.2 – 0.3 |
| Lumbar rotation (rad) | 0.51 | 0.47 – 0.55 |

Of 58 configurations, 2 crashed due to joint extrapolation beyond defined range, and 9 produced numerically invalid results (HIC15 > 10,000) confirmed by MADYMO solver warnings. The remaining 46 valid runs were used for analysis.

---

## Key findings

- Head impact speeds ranged from 3.0 to 9.1 m/s (median 5.2 m/s), spanning and exceeding the EN 1080 headform drop velocities of 4.57 and 5.42 m/s.
- All simulated impacts were oblique (20°–42° to the contact surface), compared to the EN 1080 normal (90°) impact condition.
- Peak angular accelerations exceeded the adult DAI threshold of 10,000 rad/s² in the upper quartile of the distribution, with the Y-axis (forward rotation) as the dominant loading direction.
- Peak linear accelerations and 3 ms clip values remained below the EN 1080 250 g limit across most configurations, suggesting EN 1080 linear thresholds may not be the critical failure mode for this impact scenario.

---

## Attribution

The core simulation pipeline (`simulation_functions.py`) is adapted from original code by **Lucas Lacroix** (ll6115), Imperial College London HEAD Lab. Modifications by Andria Elia include:

- CSV parser rewritten for MADYMO 2021.1 single-row output format
- HIC15 time-scaling bug fix
- Impact onset peak detection threshold added (`height=100`) to filter near-zero contact force noise
- Bicycle-van contact force extraction added
- Variable naming updated (car → van, motorcycle → bicycle)

---

## License

This code is made available for academic reference. If you use or adapt it, please cite the original thesis and acknowledge the HEAD Lab, Imperial College London.
