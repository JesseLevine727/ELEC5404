# Neural-Operator Surrogate + Generative Inverse Design for an RF Front-End Digital Twin

ELEC5404 / ELG6344 mini-project (Prof. Q.J. Zhang, Carleton University).

A neural-operator surrogate learns the map from microstrip filter geometry to
S-parameters, a knowledge-based branch injects microwave domain knowledge, a
conditional VAE inverts the surrogate for instant geometry synthesis, and the
whole chain is assembled into a front-end digital twin.

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
- [ ] MLP forward surrogate baseline
- [ ] DeepONet neural operator + frequency-extrapolation study
- [ ] KBNN knowledge-based branch + data-efficiency study
- [ ] cVAE generative inverse design + verification

Tier 2 (stretch)
- [ ] Transfer learning across filter topologies
- [ ] GAN / VAE data augmentation
- [ ] Neural space mapping to openEMS fine model
- [ ] RF front-end digital twin (filter + match + LNA cascade)
