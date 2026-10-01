# DeepRF — Deep Learning for RF/Microwave Design

ELEC5404 / ELG6344 mini-project (Prof. Q.J. Zhang, Carleton University).

Neural operators and generative models for microwave filter synthesis: a neural
operator learns the map from microstrip filter geometry to S-parameters, a
knowledge-based branch injects microwave domain knowledge, a conditional VAE
inverts the model for instant geometry synthesis, and the chain is assembled
into a front-end digital twin.

## What is this project? (plain terms)

**In one sentence:** teach a neural network to be a lightning-fast stand-in for an
electromagnetic (EM) simulator, then use it *backwards* to design microwave
filters on demand.

**The problem.** Designing a microwave filter means choosing physical dimensions
(resonator lengths, gaps, widths). To find out how a design behaves you run an EM
simulation — accurate, but slow (minutes to hours each). So design becomes
trial-and-error over hundreds of slow simulations.

**The idea.**
1. Run the simulator on many random designs → a dataset of
   `dimensions -> frequency response (S-parameters)`.
2. Train a neural network to copy that mapping; it then predicts the response in
   microseconds instead of minutes.
3. Invert it: give the network a *desired* response, and it outputs the
   dimensions that produce it — design in one shot.

Think of it as the network memorizing the simulator's "cheat sheet," so Maxwell's
equations never have to be solved again for that class of filter.

**What gets built.**

| Step | What | Status |
|---|---|---|
| 1 | Data generator — analytic filter model, dimensions -> S-parameters (4000 samples) | done |
| 2 | Forward model — dimensions -> response; MLP vs neural operator (DeepONet) | MLP done, DeepONet next |
| 3 | Knowledge-based net — bake in microwave formulas to learn from less data | planned |
| 4 | Inverse design — desired response -> dimensions, via a generative model (cVAE) | planned |
| 5 | Stretch — transfer to new filters, augment scarce data, correct against real EM, front-end digital twin | planned |

A "front-end digital twin" is a fast software replica of the whole receiver chain
(filter + amplifier + mixer) that mirrors the real hardware.

## Layout

```
config.yaml                 experiment configuration
src/
  config.py                 config loader
  data/
    filter_model.py         analytic coupled-resonator bandpass model
    generate_dataset.py     geometry -> S-parameter dataset generator
  models/                   MLP, DeepONet, KBNN, cVAE, GAN
  train/                    training entry points
  inverse/                  generative inverse design
  space_mapping/            neural space mapping to fine EM
  digital_twin/             component cascade / system-level surrogate
  utils/                    plotting, metrics
data/                       generated datasets (git-ignored)
results/                    figures and metrics
notebooks/                  analysis notebooks
```

## Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

## Generate the dataset

```bash
.venv/bin/python -m src.data.generate_dataset            # full sizes from config.yaml
.venv/bin/python -m src.data.generate_dataset --n-train 200 --n-val 50 --n-test 50
```

Produces `data/dataset.npz` with complex `S` of shape `(N, 2, 2, F)` plus the
frequency grid and geometry vectors (metres).

## Geometry vector

For an `N`-resonator filter, `geom = [L_1..L_N, S_1..S_{N-1}, S_in, S_out, W]`
(all in metres): resonator lengths, inter-resonator coupling gaps, input/output
feed gaps, and line width.

## Roadmap

Tier 1 (core)
- [x] Analytic filter model + dataset generator
- [x] MLP forward surrogate baseline
- [ ] DeepONet neural operator + frequency-extrapolation study
- [ ] KBNN knowledge-based branch + data-efficiency study
- [ ] cVAE generative inverse design + verification

Tier 2 (stretch)
- [ ] Transfer learning across filter topologies
- [ ] GAN / VAE data augmentation
- [ ] Neural space mapping to openEMS fine model
- [ ] RF front-end digital twin (filter + match + LNA cascade)
