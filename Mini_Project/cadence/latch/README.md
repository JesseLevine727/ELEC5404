# StrongARM latch, GPDK045, LVT

This is the comparator for the ELEC 5404 mini-project (MLP vs KBNN models of decision delay and energy). The design targets come from a 10-bit, 100 MS/s SAR ADC with VCM-based switching; see `../../PROJECT_PLAN.md`, section 2.

The latch is a standard StrongARM:
- input pair M1/M2
- clocked tail M0
- NMOS latch M3/M4 on the integration nodes P/Q
- PMOS latch M5/M6 on X/Y
- PMOS reset switches S1–S4
- output inverters, each driving a 5 fF load

The output polarity is OUTP = NOT X, so OUTP rises when INP > INN.

![StrongARM schematic](plots/sch_strongarm.png)

| Device | W (total) / fingers | Type |
|---|---|---|
| M1, M2 (input pair) | 12 µm / 4 | nmos1v_lvt |
| M0 (tail) | 8 µm / 4 | nmos1v_lvt |
| M3, M4 (NMOS latch) | 2 µm / 2 | nmos1v_lvt |
| M5, M6 (PMOS latch) | 2 µm / 2 | pmos1v_lvt |
| S1–S4 (reset) | 250 nm / 1 | pmos1v_lvt |
| INV45 (output) | 250 nm N / 700 nm P | LVT |

All devices have L = 45 nm. The sizes live in `skill/sa_sizes.il`.

## Files and flow

Virtuoso library: `SAL45`, in the server's `~/Desktop/StrongArm45`.

```
skill/01_build_latch.il   INV45, STRONGARM, STRONGARM_TB schematics + symbols (PDK callbacks size every device)
skill/02_netlist_tb.il    ADE/OCEAN netlist of STRONGARM_TB -> netlists/sa_tb_ade.scs (layout-aware geometry)
skill/03_screenshots.il   plots/sch_*.png (run with xvfb)
scripts/sa_sim.py         netlist composer, Spectre runner, delay/energy/losing-output measurement, width sweeps
scripts/explore_*.py      design-space explorations (approximate geometry, see below)
scripts/verify_*.py       verification on the real ADE netlist
veriloga/offset_search.va binary-search threshold finder used for offset Monte Carlo
data/                     results (json); raw/ and logs/ stay on the server
```

Steps:
1. Run once from `cadence/`: `common/cf.sh virtuoso latch 01_build_latch`, then `common/cf.sh virtuoso latch 02_netlist_tb`.
2. Copy the ADE netlist into `netlists/sa_tb_ade.scs`.
3. Run the Python scripts from `latch/`. Each script pushes, simulates and pulls through `common/cf.sh`.

**Measurements** (`sa_sim.measure`):
- **t_d:** from the clock crossing 50 % to the first output crossing VDD/2.
- **E:** VDD × supply charge over one full cycle (reset → evaluate → reset).
- **Losing-output peak:** checks for double decisions.

## How the design was reached

The `explore_*` scripts edit widths or model names in the netlist text. They do **not** recompute junction/stress geometry, so they are only good for choosing a direction. They ran on the frozen SVT first-cut netlist `netlists/sa_tb_ade_firstcut_svt.scs`. On the final design they turned out about 15 % pessimistic for t_d: 118 ps in exploration vs 102 ps with the real geometry.

1. **SVT first cut** (`01_first_look`, `02`): 451 ps at 1 mV and τ = 46 ps.
   - With 4× wider devices it was still 275 ps / 33 ps.
2. **The cause** (`03`) is two separate limits, both near the SVT threshold:
   - **Delay** is the input pair integrating slowly, because VCM = 0.5 V is close to Vth.
     - Raising VCM or using LVT input devices halves it.
   - **τ** is set by the latch devices, which regenerate around VDD/2 = 0.5 V.
     - An LVT latch gives 25 ps, all-LVT gives 14 ps.
   - **Outcome:** switched to LVT throughout.
3. **LVT sizing** (`04`, `05`):
   - The input pair dominates the delay; t_d(1 mV) floors at about 112 ps (approximate geometry) for widths ≥ 8 µm.
   - A 2 µm latch gives the best metastability slack (300 ps − t_d)/τ.
4. **Delay breakdown and output inverter** (`06`, `07`). At 1 mV:
   - X/Y reach a 10 mV split at about 47 ps.
   - Regeneration from 10 to 100 mV takes 33 ps.
   - The inverter adds the rest.
   - **Skewed inverters with a strong PMOS (1.4 µm) are not usable.** During metastability X and Y hover below their trip point, so *both* outputs rise (the losing output reaches 0.5–0.78 V). Measured τ then collapses to under 1 ps at small ΔVin and the log law breaks.
   - With a 700 nm PMOS the law holds down to 1 nV.
5. **Noise vs sizing** (`08`): widening the input pair (16/24 µm) or shrinking the tail (4 µm) left σ_n unchanged within ±7 %. It only cost speed or energy.

## Verification (ADE netlist, `verify_*.py`)

### Nominal: 27 °C, VDD 1.0 V, VCM 0.5 V (`verify_01_nominal.py`)

| Quantity | Result | Target |
|---|---|---|
| t_d at ΔVin = 1 mV | 102.1 ps | ≤ 100 ps |
| t_d at 300 mV | 46.3 ps | |
| τ (from neighbouring decades, 1 mV → 1 nV) | 9.44–9.49 ps | ≤ ~15 ps |
| Metastability slack (300 ps − t_d(1 mV)) / τ | 20.9 | |
| Energy per comparison | 30.5 fJ | ≤ 50 fJ |
| Losing-output peak | 0.25 V | |
| Symmetry | t_d(−1 mV) = t_d(+1 mV), sign correct | |

**Solver settings:**
- reltol 1e-6, vabstol 1e-12 V, iabstol 1e-13 A (`sa_sim.TOL`). These match iabstol 1e-15 within 0.002 ps over the whole sweep, and avoid a Newton failure at the clock edge that 1e-15 occasionally caused.
- Medium settings (reltol 1e-5, vabstol 1e-9, iabstol 1e-12) differ by at most 0.2 ps.

### Operating range: t_d(1 mV) / τ in ps, VCM = 0.5 V

| | 0.9 V | 1.0 V | 1.1 V |
|---|---|---|---|
| 0 °C | 130.6 / 17.1 | 97.9 / 9.1 | 82.8 / 5.7 |
| 27 °C | 134.7 / 16.9 | 102.1 / 9.5 | 86.4 / 6.1 |
| 85 °C | 147.2 / 17.5 | 113.6 / 10.5 | 96.5 / 7.1 |

**VCM effect** (27 °C, 1.0 V):

| VCM | t_d(1 mV) | τ |
|---|---|---|
| 0.4 V | 200 ps | 13.3 ps |
| 0.6 V | 81 ps | 8.9 ps |

τ depends on VCM because the input pair still carries the latch current during regeneration.

Energy over the whole range is 23–40 fJ.

Slack is 8.7 at the worst point (0.9 V, 85 °C, VCM 0.5) versus 20.9 nominal. At 100 MS/s the metastability rate goes from about 1e-9 to about 2e-4 per conversion.

### Offset (`verify_02_offset.py`)

**Method:**
- 200-sample Monte Carlo with `variations=mismatch` (GPDK045 `section=mc`).
- In each sample, `offset_search.va` binary-searches ΔVin over 13 clock cycles, to 4 µV resolution.

**Result:** **σ = 2.27 mV** (95 % interval 2.07–2.52 mV), mean −0.33 mV, range −6.6 … +5.5 mV. The target is ≤ 5 mV.

**Hand check:** the PDK's LVT mismatch is σ(Vt) = 1.78 mV·µm × 0.689 / √(WL) per device. For the 12 µm × 45 nm pair that gives 2.36 mV, so the input pair dominates.

### Noise (`verify_03_noise.py`, `explore_08_noise.py`)

**Method:** Spectre transient noise with ΔVin held fixed and 300–400 clock cycles per point. P(+) is fitted to Φ(ΔVin/σ).

**Result:**
- **σ_n = 0.60 mV rms** at 100 GHz noise bandwidth (95 % interval 0.55–0.66 mV), pooled over 3300 decisions.
- At 200 GHz: 0.65 mV (0.51–0.89 mV), consistent within error.

**Target:** ≤ 0.5 mV. This is **missed**; see the open decision in `PROJECT_PLAN.md`, section 12.

## Spectre notes

- **Multiple `options` statements do not apply per analysis.** reltol, vabstol and iabstol are global and the last value wins (warnings SFE-1131/884). Use one netlist per tolerance set. `alter` statements do apply in sequence.
- **Nested `sweep` blocks:** each closing brace must be on its own line.
- **Monte Carlo mismatch:** GPDK045 mismatch is per instance through `rn1_*`/`rn2_*` in `stat_mis_*`. It is active with `section=mc` and `montecarlo variations=mismatch`.
- **Output size:** transient-noise and Monte Carlo runs at full time resolution are large (2.2 GB for `08`). Future long runs should save only strobe points. `cf.sh spectre` now pulls only the run it just did.

## Still to do

- Kickback onto the inputs (needs a source impedance or a capacitive DAC stand-in).
- TikZ figure of the schematic for the report.
- The data-generation pipeline: static dataset plus the extrapolation set (`PROJECT_PLAN.md`, section 4).
