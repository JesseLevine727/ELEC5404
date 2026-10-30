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
