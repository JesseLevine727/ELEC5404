# StrongARM latch dataset (MLP / KBNN models)

Generated 2026-10-30 by `scripts/gen_dataset.py` from the final latch (`netlists/sa_tb_ade.scs`, md5 a58cd80f…; design in `../../README.md`, plan in `PROJECT_PLAN.md` section 4).

## What one row is

One Spectre transient of **one comparison starting from reset**:
- GPDK045, typical models (tt), LVT devices.
- Ideal inputs held at VCM ± ΔV_in/2 from t = 0.
- Clock: 0 → VDD, 20 ps edges, rising at 200 ps; 1.2 ns evaluate, then 0.8 ns reset (2 ns transient).
- Tolerances as in the verification runs: reltol 1e-6, vabstol 1e-12, iabstol 1e-13, max step 1 ps, errpreset conservative.

A Verilog-A meter (`veriloga/sa_meter.va`) sits in series with the supply and measures each comparison inside the simulation. No waveforms are kept. On 12 pilot points it matched the waveform measurement used for verification to < 1 fs in t_d and about 1 aJ in E.

## Columns (`<set>.csv`)

| Column | Unit | Meaning |
|---|---|---|
| dvin_V | V | differential input ΔV_in = V(inp) − V(inn) |
| vcm_V | V | input common mode |
| vdd_V | V | supply (also the clock amplitude) |
| temp_C | °C | temperature |
| td_ps | ps | **decision delay**: clock crossing VDD/2 (rising) → first output inverter crossing VDD/2 (rising) |
| E_fJ | fJ | **energy per comparison**: VDD × supply charge over the whole 2 ns transient (evaluate + the reset after it) |
| sign | | +1 if OUTP won, −1 if OUTN won |
| vlose_V | V | peak of the losing output while the clock is high (low = clean decision) |
| ok | | 1 if the inputs reached the circuit as listed, the decision was correct, finished inside the evaluate phase, and the losing output never crossed VDD/2 |

Every row of every set has ok = 1.
`points_<set>.csv` holds the inputs that were simulated, in the same order.
`check_sources.json` lists which test/extrap rows the sym and tol sets repeat.

## Sets

| Set | Rows | ΔV_in | VCM, VDD, T | Sampling | Intended use |
|---|---|---|---|---|---|
| train | 2800 | 100 µV – 300 mV | 0.4–0.6 V, 0.9–1.1 V, 0–85 °C | scrambled Sobol (seed 5404), **in order** | training. Any first N rows are an evenly spread design, so the data-efficiency study can take the first N. |
| val | 600 | same | same | uniform random (seed 1) | early stopping |
| test | 600 | same | same | uniform random (seed 2) | interpolation accuracy |
| extrap | 500 | 1 – 100 µV | same | uniform random (seed 3) | never train on it: the metastable-region extrapolation test |
| slices | 306 | 1 µV – 300 mV, 6 per decade (34 values) | 9 fixed conditions, 34 rows each | grid | t_d vs ΔV_in curves; τ and t₀ fitted directly to Spectre |
| sym | 50 | −ΔV_in of 25 test + 25 extrap rows | | | symmetry check (not for training) |
| tol | 50 | 25 test rows + the 25 smallest-ΔV_in extrap rows | | | rerun at half tolerances and 0.5 ps max step (not for training) |

ΔV_in is log-uniform and positive except in sym; the other inputs are uniform.
Slice conditions (VCM, VDD, T), in row order:
1. (0.5, 1.0, 27): nominal
2. (0.4, 1.0, 27)
3. (0.6, 1.0, 27)
4. (0.5, 0.9, 27)
5. (0.5, 1.1, 27)
6. (0.5, 1.0, 0)
7. (0.5, 1.0, 85)
8. (0.4, 0.9, 85): slowest corner
9. (0.6, 1.1, 0): fastest corner

## Checks

- **Symmetry:** −ΔV_in gives the same t_d within 1.1 fs and the same E within 0.12 aJ. So the models can use |ΔV_in|; the nominal latch has no offset.
- **Numerical accuracy:** halving the tolerances and the max step moves t_d by at most 2.5 fs on test rows and 55 fs at 1 µV (0.03 %).
  - The logs show about one SPECTRE-16780 warning per six transients: Spectre relaxed the error tolerance for a single step at a clock edge. The half-tolerance check shows this doesn't affect the results.
- **Nominal point** (slices row 18: 1 mV, 0.5 V, 1.0 V, 27 °C): 105.76 ps and 32.24 fJ, the same as the verification run.
- **Log law on the slices:** t_d falls monotonically with ΔV_in everywhere. From 1 to 100 µV it fits t_d = t₀ − τ·ln ΔV_in to within 0.03 ps.

| Slice (VCM, VDD, T) | τ (ps, 1–100 µV fit) | t_d at 1 µV / 100 µV / 1 mV / 300 mV (ps) | E at 1 µV / 1 mV / 300 mV (fJ) |
|---|---|---|---|
| 0.5, 1.0, 27 | 9.99 | 174.7 / 128.7 / 105.8 / 46.0 | 37.1 / 32.2 / 31.6 |
| 0.4, 1.0, 27 | 13.95 | 310.7 / 246.4 / 213.5 / 59.4 | 35.4 / 30.1 / 28.5 |
| 0.6, 1.0, 27 | 9.36 | 146.3 / 103.2 / 81.8 / 41.9 | 39.4 / 34.7 / 33.7 |
| 0.5, 0.9, 27 | 17.90 | 260.4 / 177.9 / 136.9 / 55.2 | 30.3 / 26.6 / 26.3 |
| 0.5, 1.1, 27 | 6.45 | 135.0 / 105.3 / 90.5 / 40.5 | 44.9 / 38.4 / 37.4 |
| 0.5, 1.0, 0 | 9.58 | 167.3 / 123.2 / 101.3 / 42.8 | 36.2 / 31.7 / 31.2 |
| 0.5, 1.0, 85 | 11.09 | 194.5 / 143.4 / 118.0 / 53.2 | 38.6 / 33.3 / 32.3 |
| 0.4, 0.9, 85 | 21.61 | 399.4 / 299.9 / 250.5 / 76.9 | 29.8 / 25.6 / 24.6 |
| 0.6, 1.1, 0 | 5.32 | 96.6 / 72.1 / 60.0 / 33.6 | 46.4 / 40.5 / 39.3 |

Ranges over the main sets: t_d 36–285 ps, E 23–43 fJ. Extrap: t_d 83–345 ps.

## Notes for modeling

- **Energy rises at small ΔV_in**: about 15 % from 1 mV to 1 µV. Most likely cause: near metastability X and Y stay near mid-rail longer, so the cross-coupled pairs and the output inverters conduct (crowbar current) for longer. A pure C_eff·VDD² prior misses this, so the energy model needs a ΔV_in term.
- **Low VCM matters**: at VCM 0.4 V the input pair is close to threshold. t_d at 1 mV doubles compared with VCM 0.5 V, and stays slow up to large ΔV_in (59 ps at 300 mV).
- **Large ΔV_in departs from the log law**: above a few mV the curve flattens. That is the KBNN's correction-network region.

## What the dataset leaves out

- **Noise and mismatch:** the data is deterministic. ΔV_in stands for the total effective input at the decision (signal + offset + noise).
- **Process corners:** tt only.
- **Running-SAR conditions:** each row is a first comparison from reset with ideal input sources. In a running SAR the delay is about 1 % shorter (partly recharged tail node). With a 300 fF capacitive DAC it is about 4 % longer (kickback check in `../../README.md`).

## Reproducing

    python3 scripts/gen_dataset.py points                                  # regenerates points_*.csv (fixed seeds)
    python3 scripts/gen_dataset.py run val test extrap slices tol train sym # about 25 min of Spectre in total
    python3 scripts/gen_dataset.py collect                                 # meter output -> <set>.csv, prints the checks

Spectre time is about 0.3 s per point. The full run took 22 minutes on linap-v-02.
