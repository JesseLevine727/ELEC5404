# ANN models: MLP vs KBNN

PyTorch code for the models in `PROJECT_PLAN.md` section 5. Data: `../cadence/latch/data/dataset/` (README there).

| File | What it is |
|---|---|
| `nets.py` | data loading and scaling, batched MLP and KBNN, Adam and batched L-BFGS training, metrics, errors by region |
| `exp_01_sizes.py` | accuracy and extrapolation vs size (MLP ReLU, MLP tanh, KBNN with linear and √ correction input) |
| `exp_02_lbfgs.py` | L-BFGS (from scratch, and after 2000 Adam epochs) vs Adam |
| `exp_03_guard.py` | training-set variants for the worst-case errors at the edges of the input box |
| `exp_04_select.py`, `exp_05_depth.py` | width and depth by lowest validation error (superseded by exp_06; picks kept in `results/final_exp05.pt`) |
| `exp_06_target.py` | final sizes: the smallest network with training and validation errors ≤ 0.5 % (writes `results/final.pt`) |
| `SIZING.md` | how width and depth were chosen, step by step |
| `final_eval.py` | one-time evaluation of the frozen models (`results/final_eval.csv`) |
| `exp_07_data_eff.py`, `exp_08_cost.py` | data efficiency (frozen architectures on the first N training points); training and evaluation cost |
| `figures.py` | all report figures from the saved results (`figs/*.pdf`, PNG previews); Fig 1(b) from `../cadence/latch/data/fig1_waves.npz` |
| `tables.py` | report Tables 1–6 from the saved results: `tables/tables.tex` (booktabs, to `\input`), `tables/tables.md`, `tables/preview.tex` |
| `tikz/fig2_structures.tex` | Fig 2, MLP and KBNN structures (compiled copy in `figs/`); Fig 1(a) is `../cadence/latch/tikz/strongarm.tex` |
| `meta.py`, `exp_09_meta.py` | SAR metastability error rate (threshold sum, KBNN closed form, GPU Monte Carlo) and the Fig 9 / Table 5–6 results |
| `show_history.py` | loss vs epoch table from a saved history |
| `results/` | per-model CSV, weights (`.pt`, rebuild with `nets.rebuild`), loss histories (`_hist.npz`), logs |

**How models are trained**
- Every model object is a batch of independent networks trained together on the GPU (RTX 5080, float32): 5 seeds × 6 sizes per family.
- Full batch, so one epoch is one weight update. Loss: relative MSE over (t_d, E).
- Early stopping: each network keeps its best-validation weights.

## Results so far (2026-10-30)

Test and extrapolation errors are |% error|, mean over 5 seeds.

**exp_01, Adam 30k epochs.**

| Model | Test t_d avg / max | Test E avg / max | Extrap t_d avg / max | Extrap E avg / max |
|---|---|---|---|---|
| MLP ReLU H=40 (282 weights) | 0.53 / 5.6 | 0.16 / 0.92 | 4.2 / 16.2 | 0.59 / 2.4 |
| MLP tanh H=40 | 0.29 / 3.6 | 0.07 / 0.54 | 2.6 / 17.1 | 0.98 / 4.9 |
| KBNN h=20 (306) | 0.55 / 4.1 | 0.12 / 0.84 | 0.80 / 4.3 | 0.21 / 1.3 |
| KBNN h=80 (1206) | 0.38 / 5.1 | 0.09 / 0.69 | 0.47 / 1.95 | 0.14 / 0.92 |

- The KBNN's extrapolation error is about 5× lower; inside the training range all three are similar.
- The linear correction input beat √ at every size, so √ was dropped.
- The KBNN reaches 1 % validation error about twice as fast as the MLPs (`show_history.py`).

**exp_02, L-BFGS.**
- 3000 iterations reach about Adam's 30k-epoch accuracy: slightly better for tanh and KBNN, slightly worse for ReLU, whose loss surface is not smooth.
- Each iteration takes about 14 loss evaluations, because the batch waits for its slowest line search. Wall time is therefore similar to Adam.
- It does not change the worst-case errors.

**Where the worst errors are** (errors by region, `nets.regions`). Test t_d avg / max:

| Model | ΔV_in 0.1–1 mV | 1–10 mV | 10–100 mV | 100–300 mV | Inner box | Outer 5 % of the box |
|---|---|---|---|---|---|---|
| MLP ReLU 40 | 0.45 / 1.7 | 0.49 / 2.2 | 0.50 / 1.8 | 0.79 / 5.6 | 0.46 / 1.9 | 0.67 / 5.6 |
| MLP tanh 40 | 0.19 / 1.2 | 0.18 / 0.7 | 0.33 / 1.2 | 0.61 / 3.6 | 0.20 / 0.9 | 0.45 / 3.6 |
| KBNN 20 | 0.42 / 1.7 | 0.45 / 3.1 | 0.58 / 3.6 | 0.95 / 3.7 | 0.45 / 2.1 | 0.80 / 4.1 |

- The worst errors are at ΔV_in 100–300 mV and at the faces of the box, worst at low VCM.
- t_d bends a second time there: its slope steepens from 30 to 150 mV, from about −21 to −33 ps/decade at nominal and from −49 to −112 ps/decade at VCM 0.4 V. That is probably the large-signal regime, where the input pair steers nearly all the tail current to one side.
- These are the fastest decisions (45–75 ps), so a few ps is already several %.
- The KBNN's own weak spot: low VCM, VDD and T together at 2–30 mV.

**exp_03, training-set variants.** Worst-case test error in % (t_d / E):

| Training set | MLP ReLU 80 | MLP tanh 80 | KBNN 40 | KBNN 80 |
|---|---|---|---|---|
| base, 2800 | 4.4 / 0.72 | 3.5 / 0.48 | 4.7 / 0.60 | 5.1 / 0.69 |
| + 500 interior points (control) | 4.8 / 0.64 | 3.6 / 0.45 | 3.0 / 0.68 | 3.8 / 0.63 |
| + guard band, ΔV_in to 0.66 V | 2.3 / 1.7 | 2.4 / 2.8 | 3.0 / 1.4 | 2.7 / 1.7 |
| + guard, ΔV_in ≤ 0.3 V (VCM, VDD, T only) | 2.9 / 0.83 | 3.7 / 0.42 | 4.3 / 0.62 | 4.5 / 0.66 |
| + guard, ΔV_in ≤ 0.4 V | 2.3 / 2.4 | 2.7 / 1.8 | 2.8 / 1.5 | 3.3 / 1.9 |

- The guard band halves the worst-case t_d error only when it extends ΔV_in past 0.3 V.
- That same extension doubles to triples the worst-case E error. Above about 100 mV, E turns around and rises with ΔV_in (nominal: 30.3 fJ at 100 mV, 31.6 fJ at 300 mV, about 32 fJ near 500 mV), and fitting that regime pulls the models off at the low-VCM, large-ΔV_in corner.
- Averages get slightly worse with any guard, and the ReLU MLP's extrapolation error rises (4.2 → 5–6 %).
- A guard in VCM, VDD and T only does little. More interior data does little.

**exp_04/exp_05, lowest validation error (superseded).**
- The rule picked the largest width tried: ReLU MLP 1×640 (4482 weights), tanh MLP 2×80, KBNN 1×320.
- On noise-free data, validation error never plateaus, so the 5 % parsimony test never stopped the growth.
- The large ReLU networks overlearned: validation error was 3× the training error, against about 1.07× at 40 neurons.
- The extrapolation result was the same as below. Picks and their one-time evaluation: `results/final_exp05.pt`, `final_eval_exp05.csv`.

**Frozen models (exp_06; evaluated once with `final_eval.py`).** Target accuracy E_d = 0.5 % RMS; each family takes the smallest network whose training and validation errors are both ≤ E_d. Mean over 5 seeds, |% error|:

| Model | Weights | Train / val RMS | Test t_d avg / max | Extrap t_d avg / max | Test E avg / max | Extrap E avg / max |
|---|---|---|---|---|---|---|
| ReLU MLP 1×50 | 352 | 0.39 / 0.42 | 0.45 / 6.0 | 4.2 / 16.4 | 0.13 / 0.89 | 0.52 / 2.4 |
| tanh MLP 1×12 | 86 | 0.45 / 0.46 | 0.44 / 5.2 | 3.6 / 20.8 | 0.17 / 1.6 | 2.7 / 10.8 |
| KBNN 1×25 | 381 | 0.38 / 0.44 | 0.45 / 3.5 | 0.61 / 2.9 | 0.10 / 0.66 | 0.18 / 1.1 |

- By construction, all three are equally accurate inside the training range: 0.45 ps average t_d error.
- Below 100 µV the MLPs are off by 6.5–7.5 ps on average (32–38 ps worst case); the KBNN by 1.2 ps (7.6 ps).
- ReLU 40 and KBNN 20 are labelled overlearning only by definition: training error is just under E_d and validation error just over it, with val/train 1.05–1.11. They are not memorizing.

## Comparisons on the frozen models (`figures.py`, figures in `figs/`)

**Fig 4, t_d and E vs ΔV_in at the nine Spectre slices** (`fig4_td_slices`, `fig4b_E_slices`; representative network of each family).
- Inside the training range (100 µV–300 mV) all three curves lie on the Spectre points.
- Below 100 µV the KBNN follows the straight log-law line to 1 µV at every slice.
- The ReLU MLP keeps its last slope, which is too steep at most slices: up to 30 ps high at 1 µV.
- The tanh MLP flattens out (tanh saturates): up to 38 ps low in t_d, and its energy curve bends the wrong way.
- All three miss the drop in E at 300 mV and VCM 0.4 V, the known weak spot at the box edge.

**Fig 5, error vs ΔV_in from the test set into the extrapolation region** (`fig5_extrapolation`; all 5 seeds, bins of ⅓ decade).
- In range, all three have a mean t_d error of 0.4–0.5 ps.
- Below 100 µV the MLPs' error grows about 10× over the two decades, to 13–15 ps mean and 31–38 ps worst at 1 µV.
- The KBNN's grows slowly, to 1.6 ps mean and 7.5 ps worst.
- The KBNN's in-range worst case is its own weak spot (low VCM, VDD and T at 2–40 mV), at up to 6 ps.

**Fig 8, the KBNN's parameters vs log-law fits to Spectre** (`fig8_parameters`, values in `figs/fig8_parameters.csv`). Spectre fit on the 13 slice points from 1 to 100 µV; the KBNN never saw these.

| Slice (VCM, VDD, T) | τ Spectre / KBNN (ps) | t₀ Spectre / KBNN (ps) | s Spectre / KBNN (fJ/neper) |
|---|---|---|---|
| 0.5, 1.0, 27 (nominal) | 9.99 / 9.94 | 105.6 / 105.8 | 0.710 / 0.700 |
| 0.4, 1.0, 27 | 13.95 / 15.14 | 214.3 / 212.1 | 0.767 / 0.737 |
| 0.6, 1.0, 27 | 9.37 / 9.44 | 81.6 / 81.1 | 0.696 / 0.676 |
| 0.5, 0.9, 27 | 17.90 / 17.37 | 136.7 / 136.7 | 0.540 / 0.521 |
| 0.5, 1.1, 27 | 6.45 / 6.48 | 90.4 / 90.6 | 0.947 / 0.910 |
| 0.5, 1.0, 0 | 9.58 / 9.54 | 101.2 / 101.5 | 0.666 / 0.657 |
| 0.5, 1.0, 85 | 11.09 / 11.09 | 117.9 / 117.9 | 0.777 / 0.776 |
| 0.4, 0.9, 85 (slowest) | 21.61 / 21.26 | 250.1 / 246.3 | 0.613 / 0.614 |
| 0.6, 1.1, 0 (fastest) | 5.32 / 5.46 | 59.8 / 59.0 | 0.868 / 0.843 |

- **τ:** the KBNN's own τ(z) is within 3 % of Spectre at 8 of the 9 slices; VCM 0.4 V is the exception, +8.5 %.
- **t₀** is within 1.6 %, **s** within 4 %, and **E₀** within 0.2 % (in the CSV).
- These are the network's own internal outputs, so the KBNN reads out the latch's regeneration time constant directly. The same law fitted to the KBNN's output gives the same values (the correction term is negligible below 100 µV).
- Fitting the law to the MLPs' outputs gives the slopes they extrapolate with:
  - ReLU MLP τ: 27–67 % too high at 7 of the 9 slices, within 8 % only at the two VDD 0.9 V slices.
  - tanh MLP τ: 30 % off on average; its energy slope s ranges from 0.39 down to −0.09 against 0.54–0.95.

**Fig 6, data efficiency** (`fig6_data_efficiency`, exp_07). The frozen architectures were retrained on the first N training points (Sobol prefixes), 5 seeds each; early stopping used the full validation set. Mean t_d error in ps, mean over seeds:

| N | 50 | 100 | 200 | 400 | 800 | 1600 | 2800 |
|---|---|---|---|---|---|---|---|
| Test: ReLU MLP | 5.8 | 3.2 | 1.8 | 1.1 | 0.83 | 0.54 | 0.41 |
| Test: tanh MLP | 2.4 | 0.68 | 0.56 | 0.47 | 0.45 | 0.47 | 0.43 |
| Test: KBNN | 3.9 | 2.6 | 1.7 | 1.1 | 0.85 | 0.57 | 0.48 |
| Extrap: ReLU MLP | 13.7 | 14.3 | 9.4 | 9.1 | 8.5 | 6.6 | 7.0 |
| Extrap: tanh MLP | 11.2 | 7.6 | 6.4 | 6.0 | 5.5 | 5.9 | 4.9 |
| Extrap: KBNN | 8.1 | 4.9 | 2.7 | 1.9 | 1.5 | 1.3 | 1.1 |

- **In range, the KBNN is no more data-efficient** than the ReLU MLP; the two curves overlap. The tanh MLP needs the least data, reaching 0.7 ps with 100 points.
- **Extrapolating, the KBNN with 200 points (2.7 ps) beats both MLPs trained on all 2800 (4.9–7.0 ps).** More data barely helps the MLPs there: 2800 points give only about 2× less error than 50. The KBNN's error falls steadily with N, because its τ and t₀ get more accurate.
- Energy extrapolation is the same story: the KBNN at 200 points (0.34 %) beats the MLPs at 2800 (0.68 % and 1.6 %).
- At N = 2800 the numbers differ slightly from the frozen models (e.g. KBNN 1.09 vs 1.17 ps extrap): same seeds, but different random draws in a different batch. That is the seed-to-seed spread.

**Fig 7 and Table 4, training and evaluation cost** (`fig7_training`, exp_08; RTX 5080, float32, full-batch Adam).

| | ReLU MLP 1×50 | tanh MLP 1×12 | KBNN 1×25 |
|---|---|---|---|
| Weights | 352 | 86 | 381 |
| Time per epoch | 0.31 ms | 0.33 ms | 0.77 ms |
| 60k epochs, 1 network / 5 seeds together | 18.7 s / 18.7 s | 19.2 s / 19.8 s | 44.8 s / 46.4 s |
| Epochs to validation RMS 1 % | 1500–2000 | 3400–3500 | 1100 |
| Epochs to E_d = 0.5 % | 8200–11000 | 13000–17500 | 7700–8200 |
| Time to E_d | ≈ 3.4 s | ≈ 5.8 s | ≈ 6.0 s |
| One prediction, CPU (PyTorch call) | 9 µs | 9 µs | 45 µs |
| Per prediction in batches of 10⁵, CPU / GPU | 66 / 1.3 ns | 6.8 / 0.53 ns | 51 / 2.3 ns |
| Spectre, per point (for comparison) | ≈ 0.3 s | | |

- **The KBNN converges in the fewest epochs:** 1 % at 1100 epochs vs 1500–3500. It starts from the nominal log law, so its first epochs are already about 20 % error rather than 30 %.
- **Each KBNN epoch costs about 2.4× more,** because there are two sub-networks and the correction is evaluated twice (at ΔV_in and at 0). So its time to E_d is similar to the MLPs': 3–6 s for all three.
- One network or five seeds together take the same time: these networks are too small to keep the GPU busy, so the batch is free.
- The epochs to E_d are medians over the seeds that reached it on their own: 5 of 5 for the ReLU MLP, 3 of 5 for the others. The selection rule uses the mean over seeds, which is just under 0.5 %.
- **Cost is dominated by data, not training:** the 2800 training points took about 14 min of Spectre (≈ 0.3 s each), against seconds of training.
- **Evaluation:** a model is about 7000× (KBNN) to 30 000× (MLPs) faster than Spectre for one prediction, and about 10⁸× in GPU batches. A single prediction's time is mostly PyTorch call overhead.

## Application: SAR ADC metastability error rate (`meta.py`, `exp_09_meta.py`; Fig 9, Table 5)

**Setup** (PROJECT_PLAN.md section 7):
- 10-bit asynchronous SAR. Sampling takes 25 % of the period, and each comparison is followed by 200 ps of overhead.
- Full scale 2·VDD, VCM = VDD/2, uniform input.
- The rate is the sum over the 1023 comparison thresholds of 2·ΔV_meta / V_FS, with each model's delay curve inverted at the time left. For the KBNN this has a closed form.
- **Spectre reference:** the same calculation on delay curves simulated directly (85 points from 10 fV to VDD per condition, 0.25 ps max step), at VDD 0.9/1.0/1.1 V × 0/27/85 °C.
  - It is trusted down to 1e-10 per conversion: below about 1 pV Spectre's delay is uncertain by about 1 ps (PROJECT_PLAN.md section 7).
  - At the inputs it shares with the dataset's 1 ps-step slices, it agrees within 27 fs.

**Checks of the method:**
- **Brute-force GPU Monte Carlo** of whole conversions agrees with the threshold sum within 3 % for all three models, at rates from 3e-7 to 1e-2. The counting noise there is 6–9 %. That took 10⁹ conversions for the KBNN and 2×10⁸ for each MLP.
- **The KBNN closed form** matches the inverted delay curve within 0.03 %.

**Maximum sampling rate for 1e-9 per conversion, and each model's predicted rate where Spectre gives 1e-9** (median of 5 seeds):

| VDD, T | Spectre f_max (MS/s) | f_max error: KBNN | ReLU MLP | tanh MLP | Rate ÷ Spectre's: KBNN | ReLU MLP | tanh MLP |
|---|---|---|---|---|---|---|---|
| 0.9 V, 0 °C | 209.7 | +0.5 % | −0.7 % | +5.3 % | 0.38 | 3.6 | 0 (no errors) |
| 0.9 V, 27 °C | 209.1 | +0.4 % | −1.1 % | +5.0 % | 0.43 | 6.3 | 0 (no errors) |
| 0.9 V, 85 °C | 202.9 | +0.3 % | −0.5 % | +5.3 % | 0.57 | 2.5 | 0 (no errors) |
| 1.0 V, 0 °C | 255.3 | −0.1 % | −5.3 % | −0.7 % | 1.4 | 2.6e3 | 20 |
| 1.0 V, 27 °C | 251.6 | −0.2 % | −4.9 % | −0.1 % | 1.8 | 1.7e3 | 2 |
| 1.0 V, 85 °C | 242.2 | −0.2 % | −4.2 % | +1.1 % | 1.7 | 6.8e2 | 8.5e-5 |
| 1.1 V, 0 °C | 284.5 | +0.0 % | −5.7 % | −3.0 % | 0.85 | 1.5e4 | 8e3 |
| 1.1 V, 27 °C | 279.6 | +0.1 % | −5.5 % | −2.5 % | 0.71 | 9.9e3 | 5.1e3 |
| 1.1 V, 85 °C | 269.1 | −0.1 % | −5.0 % | −1.3 % | 1.7 | 5.6e3 | 1.1e3 |

- **The KBNN predicts the error rate within a factor of 0.4–1.8 at all 9 conditions,** and the maximum sampling rate within 0.5 %.
  - Its largest misses are at VDD 0.9 V, where its learned τ is 4–6 % low (17.8 vs 19.0 ps). An error δτ shifts the rate by about exp(slack·δτ/τ²), with the slack about 20 τ.
  - Elsewhere τ is within 1.5 % and t₀ within 0.5 ps.
- **The ReLU MLP overestimates the rate by up to 4 decades** (10³–10⁴× at 1.0–1.1 V). It extrapolates with its last slope, which is 30–70 % too steep there; in Fig 9 its curve is visibly too shallow.
- **The tanh MLP is wrong in both directions.** Its delay levels off below about 10 µV, so it predicts no errors at all at VDD 0.9 V and 10³–10⁴× too many at 1.1 V. Its seeds disagree by up to 9 decades.
- **Extrapolation-only check:** with the other 9 delays taken from Spectre, so that only the stuck comparison comes from the model, the ratios hardly change (`exp_09_meta.log`). The errors are the extrapolation's, not the in-range fit's or the 0.3 V clamp's.
- **The speed limit is robust; the error rate is not.** The cliff is so steep (about 10⁻³ per 6 MS/s) that even the MLPs place f_max within 1–6 %. A 6 % f_max error is the same as a 10⁴× rate error. A designer who sets f_s from the rate spec needs the KBNN; one who only wants the speed limit could live with any model.

**Fig 9b, design use:** the KBNN's f_max(1e-9) vs VDD at 0/27/85 °C is a continuous curve with no new simulations, and it meets the Spectre reference at all 9 points (`fig9b_fmax_vdd`).

**Table 5, cost of a metastability rate** (nominal condition; RTX 5080; Spectre time per comparison from the dataset runs):

| Route | Cost |
|---|---|
| Brute-force Spectre (1e-9 to ±10 % needs about 10¹¹ conversions, i.e. 10¹² comparisons at ≈ 0.3 s each) | ≈ 10⁴ years |
| Brute-force GPU Monte Carlo with a model (10⁹ conversions) | KBNN 43 s, ReLU MLP 26 s, tanh MLP 12 s; ≈ 1 h for 10¹¹ |
| Threshold sum with a model (delay table + 1023 thresholds) | 0.2–0.4 ms for the table, 0.17 ms per f_s |
| KBNN closed form | 5 µs per f_s |
| Threshold sum with Spectre delays (85 simulations per condition) | ≈ 40 s of Spectre per condition |

- **The last row is the honest comparison.** With the threshold sum, a Spectre reference costs only about 85 simulations per condition, so the neural model's speed is not what makes the rate computable.
- The KBNN adds three things:
  - **Continuous coverage:** the 63 conditions of Fig 9b cost nothing, against about 45 min of Spectre.
  - **Validity below Spectre's numerical floor:** rates under about 1e-10 need inputs below 0.1 pV, where the log law is the only basis.
  - **A closed-form design equation**, ln(rate) linear in the window: rate = (2 mV / V_FS)·Σ_j exp(−(t_left,j − t₀)/τ).
