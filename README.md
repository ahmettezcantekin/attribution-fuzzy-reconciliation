# Uncertainty-Aware Reconciliation of Conflicting Sources on the Probability Simplex via Choquet Aggregation over Fuzzy Measures

Reference implementation and reproducibility package for the paper of the same title.

Commercial attribution sources report conflicting per-channel conversion *credit*, and on real
campaigns there is no ground truth to adjudicate the disagreement. This repository implements a
four-layer fuzzy information-fusion framework that reconciles several biased, partially overlapping
credit reports into a single distribution on the probability simplex, together with an interpretable
uncertainty band and cross-source disagreement diagnostics.

The pipeline is: a reliability-weighted fuzzy representation of each source, an interaction-corrected
conflict measure, a Choquet integral with respect to an elicited fuzzy measure (capacity) followed by
an L1-closure onto the simplex, and a Mamdani layer that produces a conformally calibrated confidence
band. The capacity is elicited from observable source properties alone, so the method requires no
labelled outcomes.

## Repository layout

```
config/           configuration for the synthetic data generator (YAML)
src/synth/        synthetic data generator with known ground truth
src/methods/      the fuzzy reconciliation pipeline (capacity, Choquet, conflict, Mamdani)
src/baselines/    competing aggregators (Dempster–Shafer, correlation-cluster, weighted mean, …)
src/eval/         recovery metrics and uncertainty calibration
src/experiments/  experiment drivers (condition grid, ablation, sensitivity, figures)
results/          generated result tables and figures
```

## Installation (Windows / PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducing the results

```powershell
py -m src.synth.generate --config config\synthetic_default.yaml   # synthetic dataset
py -m src.methods.reconcile                                       # end-to-end recovery vs. baselines
py -m src.experiments.run_synth                                   # condition grid (3 seeds)
py -m src.experiments.figures                                     # paper figures -> results\figures\
```

The result tables and figures referenced in the paper are committed under `results/`. Large,
regenerable synthetic CSVs are not tracked and are recreated by the commands above. The proprietary
mobile-game dataset behind the Section 7.8 case study is commercial-confidential and is not included;
the quantitative claims in the paper rest entirely on the reproducible synthetic experiments.

## License and citation

Released under the MIT License (`LICENSE`). If you use this work, please cite the paper and this
repository; metadata is in `CITATION.cff`.
