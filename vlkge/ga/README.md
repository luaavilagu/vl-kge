# Modality-fusion GA for TransE

The experiment has two stages. First, train `MASGATransE` once using equal
visual/textual weights to learn relation translations. Then freeze that
checkpoint and use DEAP to maximize filtered validation MRR by evolving one
global visual weight, alpha. The textual weight is always `1-alpha`.
There are no ANN scoring layers and no relation updates during GA search.

## 1. Train the scoring model

Run from the repository root in an environment with the project dependencies
and DEAP installed (`python -m pip install deap==1.4.3` if needed).

```bash
python -m vlkge.scripts.train --config vlkge/configs/wn9_img/mas_gatranse_clip.yaml
```

The configuration saves the best validation checkpoint to
`runs/wn9_img/mas_gatranse_multimodal/seed42.pt`. Alpha remains 0.5 during
training. Test evaluation is deferred until weight selection is complete.
The checkpoint includes model settings, dataset fingerprint, entity/relation
IDs, and feature-row mappings; the GA loader verifies these before inference.
Use this updated trainer rather than an older checkpoint without that metadata.

## 2. Evolve alpha

```bash
python -m vlkge.ga.run_fusion_ga --config vlkge/configs/wn9_img/mas_gatranse_clip.yaml --checkpoint runs/wn9_img/mas_gatranse_multimodal/seed42.pt --output-dir runs/wn9_img/mas_gatranse_multimodal/ga_seed42
```

Defaults: population 30 (27 random plus alpha 0, 0.5, 1), tournament size 3,
arithmetic crossover probability 0.8, random-reset mutation probability 0.2,
2 unchanged elites, at most 30 generations, and early stopping after 10
generations without improved validation MRR. Use `--patience 0` to disable
early stopping. These settings are starting values, not tuned optima.

Use `--crossover sbx --sbx-eta 20` to compare bounded simulated binary
crossover. Both alternatives preserve alpha in [0, 1]. Mutation resets the
single gene uniformly in [0, 1]. Crossover and mutation are independent.
DEAP clones parents and invalidates changed fitness; unchanged elites retain
their fitness. HallOfFame separately records the best-ever individual.

The search caches visual/textual scores on CPU, then evaluates each genome
without further model forwards. Full split caches consume RAM proportional
to queries times entity candidates. `--candidate-batch-size` bounds temporary
model scoring tensors, not the final cache. Optional `--max-validation-triples`
and `--max-test-triples` support smoke runs; reports record these limits.

## 3. Compare and evaluate

Every search reports visual-only, textual-only, equal-weight, random-search,
and grid-search validation metrics. Random search has the same number of
fitness evaluations as the GA's unique evaluated alphas. The grid uses the
same count unless `--grid-points` explicitly changes it. Fitness requests and
unique evaluations are both recorded because unchanged/duplicate genomes
reuse fitness. Fixed baselines are reported separately from search budgets.

To add final test results after validation-based selection, append
`--evaluate-test` to the search command. Or reuse a saved full search report:

```bash
python -m vlkge.ga.run_fusion_ga --config vlkge/configs/wn9_img/mas_gatranse_clip.yaml --checkpoint runs/wn9_img/mas_gatranse_multimodal/seed42.pt --weights runs/wn9_img/mas_gatranse_multimodal/ga_seed42/fusion_result.json --evaluate-test --output-dir runs/wn9_img/mas_gatranse_multimodal/final_seed42
```

`--weights` skips evolution for the saved GA alpha; baseline searches are
recomputed using the saved GA evaluation budget. The loader requires the
original validation subset and evaluation direction; if a subset was sampled,
reuse its sampling seed as well.
Repeat with additional `--seed` values and separate output directories to
measure search variability. Choose configurations using validation data;
test performance must not feed back into operator or weight selection.

Output files:

- `fusion_result.json`: settings, provenance, history, weights, and metrics.
- `fusion_weights.json`: checkpoint reference and selected modality weights.
- `history.csv`: per-generation best/mean MRR, alpha diversity, evaluation counts.

The original checkpoint is unchanged. Load it and call `set_alpha` with the
saved visual weight to reproduce the fused scorer. The GA finds the best
weight it encounters, not a guaranteed global optimum.

## Evaluation rules

All entities are candidates. Head and tail queries have separate all-known
filter maps; other known true answers are excluded and each query's own
target is retained. Ties use ascending entity ID, equivalent to stable sorting
scores over ascending entity candidates. The dedicated GA training path uses
the same directional maps. Older model paths keep their prior filtering.
Report MRR, Hits@1/3/10, and per-relation metrics. No binary probability
threshold is applied to raw TransE scores.

## Code guide

- `modality_fusion_ga.py`: DEAP types, operators, population, elitist evolution.
- `evaluation.py`: direction-specific filtering, fixed score cache, metrics,
  and baseline comparisons.
- `run_fusion_ga.py`: configuration/checkpoint loading, freezing, CLI, reports.
- `../models/mas_gatranse.py`: differentiable scoring model and alpha buffer.

Each function contains a docstring explaining its role and assumptions.
Refer to the [DEAP tools](https://deap.readthedocs.io/en/master/api/tools.html)
and [variation algorithms](https://deap.readthedocs.io/en/master/api/algo.html)
for the library behavior used here.

## 4. Run the agents with GA-selected fusion

The original `vlkge.mas.run_mas` command still targets MASNeuralTransE.
Use the dedicated GA agent runner after training and alpha search:

```bash
python -m vlkge.mas.run_ga_mas --config vlkge/configs/wn9_img/mas_gatranse_clip.yaml --checkpoint runs/wn9_img/mas_gatranse_multimodal/seed42.pt --fusion-weights runs/wn9_img/mas_gatranse_multimodal/ga_seed42/fusion_weights.json --head n11669921 --relation _hyponym --tail n11992806
```

The runner verifies that the weights belong to the checkpoint, loads/fixes
alpha, and runs VisualAgent, TextualAgent, then DecisorAgent. Both evidence
agents keep their separate modality messages. The decision agent delegates
aggregation through EvidenceAdapter to the model's `fuse_evidence_scores`
method, also used by direct `forward()` calls. It therefore uses the selected
alpha rather than silently reverting to equal weights. There is no GA search
or relation training during an agent cycle.

Output includes the selected weights, checkpoint fingerprint, raw modality
scores, and weighted combined score. The default decision is `unclassified`
because the search optimizes ranking MRR, not a binary probability cutoff.
If a threshold has been chosen separately on validation data, pass
`--decision-threshold VALUE` to request accepted/rejected decisions. Existing
ANN environment callers keep their previous averaging and threshold defaults.

The integration checks compare agent scores to model forward scores and
cached GA ranking scores. This numerical agreement does not constitute a
full WN9-IMG experiment or validate the starting GA hyperparameters as optimal.
