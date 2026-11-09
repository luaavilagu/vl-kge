# Three-modality GA fusion for TransE

Train a scoring model once, freeze it, then use DEAP to learn the relative
contributions of visual, textual, and structural evidence. There are no ANN
scoring heads and no relation/embedding updates during the genetic search.

## Model and weights

The fixed modality order is **visual, textual, structural**. An individual
contains three nonnegative weights summing to one, with two independent
degrees of freedom:

`score = w_visual * S_visual + w_textual * S_textual + w_structural * S_structural`

Each modality score is margin minus its TransE translation distance.
Structural entity embeddings and shared relation vectors learn during the
initial training stage; supplied visual/textual feature tables stay frozen.
The initial fusion is equal thirds. Model `modality_weights` is a buffer,
saved in state_dict and excluded from gradient optimization. `set_weights`
updates it in place, and `fuse_evidence_scores` is shared with the agents.

## 1. Train with the existing base configuration

Run from the repository root. No additional YAML configuration is created.
CLI arguments override the base YAML; the checkpoint stores the effective
arguments, dataset fingerprint, entity/relation IDs, and feature-row mappings.

```bash
mkdir -p runs/wn9_img/mas_gatranse_3modalities

nohup python -m vlkge.scripts.train \
  --config vlkge/configs/wn9_img/transe_clip.yaml \
  --model MASGATransE \
  --fusion_mode average \
  --use_structural \
  --use_visual \
  --use_textual \
  --seed 42 \
  --skip_final_test \
  --save_path runs/wn9_img/mas_gatranse_3modalities/mas_gatranse_average_3modalities_seed42.pt \
  > runs/wn9_img/mas_gatranse_3modalities/nohup.out 2>&1 &
```

The existing training helper learns structural entities and relations with
equal thirds. `--skip_final_test` reserves final test metrics for after fusion
selection. A new checkpoint is required: old two-modality checkpoints do not
contain the trained structural table or the new weight-buffer schema.
The existing two-modality YAML and experiment files are left unchanged.

The GA loader uses effective arguments from the checkpoint, so
`model: TransE` in the base YAML does not undo `--model MASGATransE`.
Existing saved data/feature paths are preferred; when those paths are missing
after moving a checkpoint, the supplied base YAML provides fallback locations.
Dataset hashes and feature/vocabulary ordering must still match.

## 2. Evolve the relative contributions

```bash
python -m vlkge.ga.run_fusion_ga \
  --config vlkge/configs/wn9_img/transe_clip.yaml \
  --checkpoint runs/wn9_img/mas_gatranse_3modalities/mas_gatranse_average_3modalities_seed42.pt \
  --output-dir runs/wn9_img/mas_gatranse_3modalities/ga_seed42
```

Starting settings: population 30, tournament size 3, arithmetic crossover
probability 0.8, random-reset mutation probability 0.2, 2 unchanged elites,
maximum 30 generations, and early stopping after 10 generations without
improved MRR. These defaults have not been tuned. Use `--patience 0` to run
the full generation limit, and repeat with separate `--seed`/output paths.

The initial population contains equal thirds and the three single-modality
vertices, plus 26 random vectors. Random initialization and reset mutation
sample uniformly over the valid weight triangle (Dirichlet(1,1,1)).
Arithmetic crossover uses one coefficient for the whole vector, preserving
its sum. Optional `--crossover sbx --sbx-eta 20` bounds coordinates and repairs
the resulting sum. Crossover and mutation are independent stages.
DEAP clones parents and invalidates changed fitness; explicit elite copying
and HallOfFame preserve the best fitness encountered.

Scores are cached in visual/textual/structural order. Each genome recombines
those fixed scores; no model forwards or parameter training occur per genome.
The score tensor uses 50% more storage than the previous two-channel cache.
`--candidate-batch-size` bounds scoring intermediates, not the full CPU cache.
Optional `--max-validation-triples` and `--max-test-triples` are recorded smoke
limits; omit them for full evaluation.

## 3. Comparisons, provenance, and test evaluation

Report equal thirds, visual-only, textual-only, structural-only, random
search, and simplex-grid search. Random search includes the same four
references and matches the GA's unique fitness-evaluation budget.
A grid uses the largest complete triangular lattice within that budget or
`--grid-points`; its actual count is reported and may be smaller than the cap.
For example, `--grid-points 231` evaluates a lattice with 0.05 increments.
Equal thirds is also reported separately, even if absent from the lattice.

Candidate filtering is directional. GA/agent validation ranking uses known
training + validation triples, retaining the current target; final test uses
training + validation + test triples. All entities are candidates, with stable
entity-ID tie breaking. Recompute comparison baselines under this protocol.
The existing training helper's checkpoint-selection evaluation still uses
its pre-existing all-known filters; its logged MRR and older ANN/GA metrics
may therefore differ from freshly recomputed validation metrics.

Use `--evaluate-test` after selecting the search configuration on validation.
To reuse an existing selected vector without repeating evolution, pass
`--weights PATH_TO_FUSION_RESULT_JSON`. Reuse the original direction/subset
and sampling seed. Baseline searches are recomputed with the saved budget.

Outputs:
- `fusion_result.json`: version 2, modality order, named weights, provenance,
  search configuration/history, and comparison metrics.
- `fusion_weights.json`: small versioned weights file tied to its checkpoint.
- `history.csv`: best/mean MRR, each best modality weight, vector diversity,
  fitness requests, and unique evaluations.

Old two-modality checkpoints/weight files are rejected clearly. Archived
summary JSON files can still be displayed by the report formatter.

## 4. Run the four agents for one triple

```bash
python -m vlkge.mas.run_ga_mas \
  --config vlkge/configs/wn9_img/transe_clip.yaml \
  --checkpoint runs/wn9_img/mas_gatranse_3modalities/mas_gatranse_average_3modalities_seed42.pt \
  --fusion-weights runs/wn9_img/mas_gatranse_3modalities/ga_seed42/fusion_weights.json \
  --head n11669921 --relation _hyponym --tail n11992806
```

VisualAgent, TextualAgent, and StructuralAgent emit separate evidence.
DecisorAgent delegates to the same weighted fusion as model forward().
The decision threshold defaults to 0.5 for the requested ANN-style test;
accepted/rejected output does not affect MRR or weight selection.

## 5. Evaluate agents on every validation triple

```bash
python -m vlkge.mas.evaluate_ga_mas \
  --config vlkge/configs/wn9_img/transe_clip.yaml \
  --checkpoint runs/wn9_img/mas_gatranse_3modalities/mas_gatranse_average_3modalities_seed42.pt \
  --fusion-weights runs/wn9_img/mas_gatranse_3modalities/ga_seed42/fusion_weights.json \
  --split val \
  --output-dir runs/wn9_img/mas_gatranse_3modalities/agents_validation
```

Each observed triple executes four agents and checks the combined score
against direct inference. Ranking uses cached replacement-candidate model
scores with the same shared fusion, rather than millions of Mesa cycles.

Outputs:
- `agent_scores.csv`: all three evidence scores, combined score, and decision.
- `agent_messages.jsonl`: four messages per observed triple.
- `summary.json`: named weights, effective buffer values, metadata, parity
  check, MRR/Hits@K, and equal-thirds comparison.
- `results.txt`: ANN-style overall and named per-relation reports.

Use `--scores-only` to skip ranking, `--split test` for final test evaluation,
and omit `--max-triples` to process the full split. Display saved metrics:

```bash
python -m vlkge.mas.evaluate_ga_mas \
  --summary runs/wn9_img/mas_gatranse_3modalities/agents_validation/summary.json \
  --report both
```

The two report blocks show equal thirds and the selected three-weight vector.
`--report equal` or `--report best` selects one block. Formatting preserves
four decimals overall and three decimals per relation without recomputation.

## Code and checks

`modality_fusion_ga.py` owns DEAP/operators; `evaluation.py` owns cached
ranking and baselines; `run_fusion_ga.py` owns checkpoint/CLI/report handling.
The model and agent adapter share the actual fusion operation. Functions
contain docstrings explaining invariants and assumptions.

```bash
python check_mas_gatranse.py
python -m unittest discover -s tests -v
```

The suite includes structural gradients, invalid weights, seeded operators,
agent/cache/model agreement, CLI-overridden training, metadata checks, and a
tiny end-to-end workflow. It is not a full WN9-IMG performance benchmark.
