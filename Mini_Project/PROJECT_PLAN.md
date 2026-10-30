# ELEC 5404 Mini-Project: Project Plan

**Working title:** *Knowledge-based neural model of a GPDK045 StrongARM comparator for SAR ADC metastability analysis*

**Course:** ELEC 5404 / ELG 6344, Neural Networks for High-Speed/High-Frequency Circuit Design (Prof. Q.J. Zhang)
**Student:** Jesse Levine
**Category:** 3, application of ANN to a practical example. The MLP-vs-KBNN study supplies the category-2 comparison.
**Due:** end of semester (date to confirm)
**Plan written:** 2026-10-29

---

## 1. Summary

A StrongARM latch is the comparator in most SAR ADCs. How long it takes to decide grows without bound as its input difference shrinks; that is metastability. An ADC's metastability error rate (about 10⁻⁶ to 10⁻⁹ per conversion) cannot be measured in transistor-level simulation, because it would take billions of comparisons.

This project builds fast neural models of a fresh GPDK045 StrongARM latch from Cadence Spectre data. The models predict the latch's **decision delay** and **energy per comparison** from its input difference and operating conditions. Two structures are compared:

- **MLP:** a pure black box.
- **KBNN:** a knowledge-based network with the regeneration law t_d = t₀ + τ·ln(1/ΔV_in) built in.

They are compared on accuracy, extrapolation into the metastable region, data efficiency, training cost and interpretability. The better model is then used to predict the metastability error rate of a 10-bit SAR ADC versus sampling rate, supply and temperature, and those predictions are checked against direct Spectre simulations.

---

## 2. Target SAR ADC and comparator specs (decided 2026-10-29)

**Target ADC:** 10-bit asynchronous SAR at **100 MS/s nominal**, GPDK045, VDD = 1.0 V. Top-plate sampling with a 2 Vpp differential full scale, so 1 LSB ≈ 1.95 mV.
**VCM-based switching**, so the comparator input common mode is fixed at **0.5 V**.
**Low-threshold (LVT) devices** throughout (`nmos1v_lvt` / `pmos1v_lvt`). Temperature range **0–85 °C**.

The first SVT build missed the speed specs badly: about 450 ps at 1 mV and τ ≈ 46 ps; with 4× devices still about 275 ps and 33 ps.
There are two separate limits:
- **Delay:** the input pair integrates slowly because VCM = 0.5 V is near the SVT threshold.
- **τ:** the cross-coupled latch sits at VDD/2 during regeneration, also near threshold.

Low-threshold devices throughout gave τ ≈ 14 ps at first-cut sizes. Changed to LVT on 2026-10-29.

| Comparator spec | Target | Reason |
|---|---|---|
| Decision time at ΔV_in ≈ ½ LSB (1 mV) | ≤ 100 ps | per-bit cycle ≈ 800 ps; the comparator gets about 300 ps, the rest goes to logic and DAC settling |
| Regeneration time constant τ | ≤ ~15 ps | near metastability the latch sits at VDD/2; SVT is too close to threshold there (≈ 33–46 ps measured), LVT reaches ≈ 14 ps |
| Metastability rate at 100 MS/s | < 10⁻⁶ per conversion | follows from a ~300 ps budget and τ ≈ 15 ps (metastable below about 0.4 nV) |
| Input-referred noise | ≤ 0.5 mV rms (≈ LSB/4) | costs at most about 0.3 bit of ENOB; sets the input-pair gm and integration |
| Offset (mismatch, Monte Carlo) | ≤ ~5 mV, 1σ | static offset only becomes ADC offset (calibratable); not modeled |
| Energy per comparison | ≤ ~50 fJ | about 11 decisions × 100 MS/s ≈ 50 µW |
| Output load | output inverters + ~5 fF per side | stands in for the SAR logic |
| Clock | 20 ps edges, reset while the clock is low | |

**Verified (2026-10-29, nominal 27 °C, 1.0 V, VCM 0.5 V):**
- t_d(1 mV) = 102 ps
- τ = 9.5 ps, constant from 1 mV down to 1 nV
- E = 30 fJ
- offset σ = 2.27 mV (200-sample Monte Carlo)
- noise σ = 0.60 mV rms (transient noise)

Noise misses the 0.5 mV target. Widening the input pair or shrinking the tail did not change it, so the spec is an open decision (section 12).
Across the dataset range τ varies from 5.5 ps (1.1 V) to about 17.5 ps (0.9 V, 85 °C).

The 100 MS/s choice comes from the metastability "cliff". Assuming 80 ps delay at 1 mV and τ = 15 ps, the rate is about 4×10⁻⁷ per conversion at 100 MS/s but about 10⁻² at 200 MS/s. The application (section 7) sweeps fs from 50 to 400 MS/s across that cliff.

---

## 3. Circuit (Cadence Virtuoso / Spectre, GPDK045)

- **Topology:** standard StrongARM.
  - Clocked tail, input pair, cross-coupled NMOS and PMOS pairs.
  - Four reset PMOS switches, on the integration nodes and the outputs.
  - Output inverters driving a representative SAR-logic load.
- **Sizing (done 2026-10-29):** by simulation sweeps on the ADE netlist, then rebuilt and verified with the layout-aware geometry the PDK callbacks netlist (including LOD stress parameters). GPDK045 bin edges were checked first and are benign.
  - **Sizes (all L = 45 nm, LVT):**
    - input pair 12 µm (4 fingers)
    - tail 8 µm (4 fingers)
    - NMOS and PMOS latch 2 µm (2 fingers)
    - reset switches 250 nm
    - output inverters 250 nm / 700 nm
  - **Details:** `cadence/latch/README.md`.
- **Testbench:**
  - ideal clock
  - differential input VCM ± ΔV_in/2
  - one comparison per simulation (fresh reset, so no memory between decisions)
- **Measured quantities:**
  - **t_d:** clock 50 % crossing to the winning output crossing VDD/2. The losing output stays below 0.25 V, so this is effectively the differential crossing.
  - **E:** VDD × supply charge over one full reset + evaluate cycle.
- **Spec verification** (not modeled): delay and τ, energy, noise (transient noise), offset (Monte Carlo) are done; kickback is still to do.

---

## 4. What the models predict

| | Variable | Range | Sampling |
|---|---|---|---|
| **Inputs** | \|ΔV_in\| (differential input) | 1 µV – 300 mV | log-uniform |
| | VCM | 0.4 – 0.6 V (nominal 0.5 V) | uniform |
| | VDD | 0.9 – 1.1 V | uniform |
| | Temperature | 0 – 85 °C | uniform |
| **Outputs** | t_d, decision delay | ~tens – hundreds of ps | |
| | E, energy per comparison | ~tens of fJ | |

The nominal (no-mismatch) latch is symmetric, so the model uses |ΔV_in|. Both signs are simulated once as a symmetry check.

### Datasets

| Set | ΔV_in range | Size (approx.) | Use |
|---|---|---|---|
| Training / validation / test | 100 µV – 300 mV | ~4000 (70 / 15 / 15 %) | training, early stopping, interpolation test |
| Extrapolation | 1 µV – 100 µV | ~500 | never trained on; tests the metastable region |
| Spot checks | specific points | ~20–50 | application validation |

---

## 5. Models

**Common to both:**
- **Input scaling:** log₁₀|ΔV_in|, VCM, VDD and T scaled to [−1, 1].
- **Output scaling:** outputs scaled too; the loss is MSE on scaled outputs.

**MLP (black box)**
- Inputs (log₁₀|ΔV_in|, VCM, VDD, T) → outputs (t_d, E).
- 1–2 hidden layers with sigmoid/tanh activations.
- Hidden-layer size swept, e.g. 10, 20, 40, 80 neurons, to address "how many hidden neurons" and to compare against the KBNN at similar parameter counts.

**KBNN (knowledge-based)**

    t_d = t₀(VCM, VDD, T) + τ(VCM, VDD, T) · ln(V_ref / |ΔV_in|) + g(all inputs)

- t₀ and τ come from small sub-networks; τ is kept positive with a softplus.
- g is a small correction network for the large-ΔV_in regime, where regeneration is no longer small-signal and the log law flattens.
- Energy has a weak prior, E = C_eff(VCM, VDD, T)·VDD² + h(all inputs), or uses a plain network. To decide.
- Optional extra variants from the textbook's knowledge-based chapter, if time allows: the difference method (the network learns the residual from the formula) and prior-knowledge input (the formula's output is fed in as an extra input).

**Training procedure (identical for all models)**
- Same data splits and scaling.
- Adam, with a short learning-rate tune per model. A quasi-Newton optimizer (L-BFGS) is optional, as a training-method comparison.
- Early stopping on the validation set.
- **Five random seeds** per configuration; report mean ± spread.
- Errors reported in physical units: % error in t_d and E, both the course's average and worst-case test error.

---

## 6. Comparisons (MLP vs KBNN)

| # | Comparison | Metric | Shown in |
|---|---|---|---|
| 1 | Accuracy on held-out data | average and worst-case % error, t_d and E | Table 3, Fig 3 (optional) |
| 2 | Fit to the physics curves | t_d vs ΔV_in overlays at several VCM / VDD / T slices | Fig 4 |
| 3 | **Extrapolation toward metastability** (key result) | error (ps) for 1–100 µV, never trained on | Fig 5 |
| 4 | Data efficiency | test error vs number of training samples (50 → 3000) | Fig 6 |
| 5 | Training cost | loss per epoch; epochs and wall-clock time to a target error; evaluation time per prediction | Fig 7, Table 4 |
| 6 | Interpretability (KBNN) | learned τ and t₀ vs values fitted directly to Spectre | Fig 8 (optional) |

---

## 7. Application: SAR ADC metastability error rate

A decision is metastable if it is not finished within the per-bit time budget T_budget, which follows from f_s, the sampling time and N. In the KBNN, the input difference at which that happens has a closed form:

    ΔV_meta = V_ref · exp( −(T_budget − t₀) / τ )

For a uniformly distributed input, the probability of a metastable event per conversion is about 2·ΔV_meta / V_LSB. That is dominated by the LSB decisions; the catastrophic MSB events are about 2·ΔV_meta / V_FS. These estimates are to be refined.

- **Output:** metastability error rate vs sampling rate, VDD and temperature, as predicted by the MLP and by the KBNN (Fig 9).
- **Validation:** direct Spectre simulations at tiny ΔV_in show which model is right.
- **Speed:** Spectre time per decision vs model time per evaluation, and what 10⁹ decisions would cost each way (Table 5).

---

## 8. Figures and tables

**Core**
- **Fig 1:** StrongARM schematic (Virtuoso and TikZ), plus transient waveforms at several ΔV_in showing how t_d is defined and how delay grows.
- **Fig 2:** MLP and KBNN structure diagrams.
- **Fig 4:** t_d vs ΔV_in, Spectre against both models, at several condition slices.
- **Fig 5:** extrapolation into the metastable region, with the untrained region shaded and an error panel.
- **Fig 6:** test error vs training-set size (interpolation and extrapolation), mean ± spread.
- **Fig 7:** training and validation loss curves per epoch, mean ± band.
- **Fig 9:** SAR metastability error rate vs sampling rate, MLP vs KBNN vs Spectre checks.
- **Table 1:** input/output variables, ranges, sampling, dataset sizes.
- **Table 2:** model structures (layers, neurons, parameter counts).
- **Table 3:** training and test errors (average and worst-case) for t_d and E.
- **Table 4:** training cost (epochs, seconds, time to target error) and evaluation time.

**Optional**
- **Fig 3:** parity plots (predicted vs Spectre).
- **Fig 8:** KBNN-learned τ and t₀ vs directly fitted values.
- **Table 5:** speedup over Spectre.

---

## 9. Timeline (to end of semester)

| Week | Dates (approx.) | Work |
|---|---|---|
| 0 | Oct 29 – Nov 1 | This plan; project setup; confirm the SAR target and the AI-use policy |
| 1 | Nov 2 – 8 | Final specs; GPDK045 gm/Id tables and bin check; latch design and testbench; spec verification |
| 2 | Nov 9 – 15 | Data-generation pipeline; main and extrapolation datasets; data quality checks (symmetry, monotonicity, numerical floor) |
| 3 | Nov 16 – 22 | MLP baseline and size sweep; KBNN; training procedure |
| 4 | Nov 23 – 29 | Comparisons 1–6; figures and tables |
| 5 | Nov 30 – Dec 6 | SAR metastability application; Spectre spot checks; report draft |
| buffer | Dec 7 – due date | Polish and submit |

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Spectre's numerical floor at tiny ΔV_in: tolerance noise, not ΔV_in, decides the result | Tight tolerances and small maximum time step; confirm results don't move when tolerances are halved; check ±ΔV_in symmetry and that t_d keeps increasing down to 1 µV |
| Memory between decisions (incomplete reset) | One comparison per simulation with a full reset; static dataset only |
| Large-ΔV_in regime departs from the log law | KBNN correction network g; report error by ΔV_in region |
| Mismatch offset not in the model | State it as a limitation; Monte Carlo offset is used only for spec verification |
| GPDK045 model bins and layout-dependent effects | Check bin edges; keep finger widths inside one bin; size with the netlisted (CDF) geometry |
| Scope creep | Core figures and tables first; optional items only if ahead of schedule |

---

## 11. Roles and tools

- **Cadence side:** latch schematic, testbenches, data generation and validation simulations, through Virtuoso / OCEAN / Spectre on `linux.doe.carleton.ca`. Built with AI assistance (Claude).
- **ANN side:** model design, training, comparisons, analysis and the report, by Jesse. Python / PyTorch.
- **To do:** confirm with Prof. Zhang what AI assistance is permitted, and acknowledge it in the report.

---

## 12. Open decisions

- [x] SAR target: 10-bit, **100 MS/s**, 1 V, VCM-based switching (VCM = 0.5 V), **LVT devices** (SVT first, changed to LVT after the first build; 2026-10-29).
- [x] Temperature range: **0–85 °C**. Process corners are not a model input (nominal tt only).
- [ ] Comparator noise spec: measured 0.60 mV rms vs the 0.5 mV target (≈ 0.55 vs 0.42 bit of ENOB). Either accept about 0.65 mV (LSB/3), or add X/Y capacitance, which costs speed and τ.
- [ ] Energy model in the KBNN: weak C_eff·VDD² prior or a plain network.
- [ ] Optional KBNN variants (difference method, prior-knowledge input): include or not.
- [ ] Optimizer comparison (Adam vs L-BFGS): include or not.
