---
marp: true
theme: default
paginate: true
title: harbench — Candidate Selection & User-Split Methodology
---

# `harbench` — Multi-Candidate Selection & User Splitting
### Methodology-grade walkthrough: the equations behind "is this candidate selected", and how train/val/held-out users are chosen

Scope of this deck (per request): not a full repo tour (see
`slide/finetune_overview.md` for that) — this is the **selection cascade**
that decides which baseline-dataset subset+weight gets finetuned into a
target-dataset model, and the **user-role assignment** (train / val /
held-out) that makes every reported number honest. Written to be liftable,
almost verbatim, into a PerCom-style methodology section.

---

## 1. Goal, and where this sits in the pipeline

**Question this material answers**: given an unseen target HAR dataset,
which subset `S` of baseline datasets and per-baseline mixture weight `w`
should a pretrained SSL backbone be finetuned on, and how is "the winner"
mathematically determined — while guaranteeing the reported number was never
seen by any search decision?

```
lightweight-score-unify   heavyweight-similar-score
  (signal-quality gate)     (target-similarity rank)
        │                          │
        └───────────┬──────────────┘
                     ▼
     optimal_subset_selection  (sibling repo, orchestrator)
   semantic_scorer → features → candidate_pool/llm_proposer
        → surrogate (GP+UCB) → reward_combined → winner
                     │  shells out per candidate, real trial
                     ▼
        harbench/finetune.py  (THIS REPO)
   --loss_mode ce_scl pooled trial   (search-time, many times)
   run_finetune_multi_candidate      (confirmatory, once)
                     │
                     ▼
        held-out honesty check (never searched over)
```

`harbench` is the **execution engine**: it never decides which candidate to
try next or which one wins — that logic lives in the sibling repo
`optimal_subset_selection`. `harbench` is trusted to (a) train a specific
manifest correctly and (b) enforce the train/val/test **user** partitions
that make every number it returns honest. This deck documents both halves —
the upstream decision math (cross-repo, read from source, not imported) and
`harbench`'s own splitting code — because a methodology section needs both
to describe the experiment precisely.

---

## 2. The selection cascade — every equation, in pipeline order

Six independent scores/decisions compose into "is (S, w) selected as the
next trial" and "is trial t selected as the winner." Each stage below names
its owning repo — only the last two run inside `harbench` itself.

### Stage A — Signal-quality gate (repo: `lightweight-score-unify`)

Per `(user, sensor-location)` row `r`, for each metric `m` with fixed
absolute anchors `(bad_m, good_m)` (no peer comparison):

```
score_m(r)              = clip( (value_m(r) - bad_m) / (good_m - bad_m), 0, 1 )
unify_lightweight_score(r) = mean_m score_m(r)
```
(`lightweight_check.py:204-209`)

Dataset-level score is the mean over every row:
```
unified_lightweight_score(D) = mean_r unify_lightweight_score(r)
```
(`lightweight_check.py:224-232`)

**Gate** (over the historical population `P` of every previously-scored
dataset's `unified_lightweight_score`, `μ = mean(P)`, `σ = std(P)`):
```
D passes  ⟺  unified_lightweight_score(D) ≥ μ − 3σ
```
(`lightweight_check.py:411-431` — "a dataset fails if its score falls below
mean − 3·stdev of historically scored datasets", a one-tailed 3σ outlier
rule, not a fixed absolute cutoff). `harbench`'s side of the handoff only
*reads* this gate's pass/fail JSON as data
(`optimal_subset_selection/lightweight_pool.py:37-46`); it never recomputes
or imports the scoring code — a **cross-repo, read-as-data boundary**.

---

### Stage B — Target-similarity re-scoring (repo: `heavyweight-similar-score`)

For each baseline `i` that passed Stage A, against the fixed target `T`,
pairwise `silhouette(T,i)` and `mmd(T,i)` are computed on L2-normalized
512-d embeddings. Each metric is **leave-one-out z-scored** against the
*other* passed baselines' own pairwise values (not against itself):

```
z_i   = ( x_i − mean_{j≠i}(x_j) ) / std_{j≠i}(x_j)
low_clip:  s_i = ( clip(−z_i, −σ, σ) + σ ) / (2σ)         (σ = config.ZSCORE_SIGMA)
```
(`heavyweight-similar-score/scoring.py:4-6,31-46,93-98` — both `silhouette`
and `mmd` use the `low_clip` treatment in this repo's convention)

```
heavyweight_score(T, i) = mean( s_silhouette(i), s_mmd(i) )
```
(`heavyweight-similar-score/scoring.py:101-119` — "`only_sh_mmd` branch":
two other metrics this module defines, `effective_rank`/`temporal_coherence`,
are deliberately unhooked from the score)

`harbench`'s pipeline neighbor consumes this as a pre-rendered markdown rank
table (`heavyweight_scores.parse_rank_md`), never re-derives it.

---

### Stage C — Semantic compatibility (repo: `optimal_subset_selection`, LLM)

```
activity_semantic_score(i)        ∈ [0,1]
placement_semantic_score(i, p)    ∈ [0,1]   (per baseline_placement p)
```
Not a closed-form equation: one forced-tool-use Claude call per target run
(`semantic_scorer.py:180-279`, model `config.MODEL_SEMANTIC_SCORER`),
clipped to `[0,1]` (`_clip01`, `semantic_scorer.py:176-177`). The *inputs*
handed to the LLM are exact, not semantic: `shared_activities` /
`target_activities_uncovered` are plain set intersection/difference over
taxonomy-canonicalized activity labels (`_shared_and_uncovered`,
`semantic_scorer.py:111-117`) — the LLM is asked only for the compatibility
judgment those sets can't answer by themselves (e.g. "walking" on a
treadmill vs. outdoors).

### Stage D — Empirical baseline score (repo: `optimal_subset_selection`)

```
empirical_baseline_score(i) = ( 1 / |{t : i ∈ S_t}| ) · Σ_{t : i ∈ S_t} macro_f1(t)
```
over every **real trial** `t` run so far in *either* phase
(`features.py:60-76`). Falls back to the neutral constant `0.5` if baseline
`i` has never yet appeared in a trial (`features.py:38,90-91,117-119`). This
is the one feature that can only reflect a baseline's behavior *after* real
trials exist — everything above is frozen before the first trial runs.

### Stage E — Portfolio feature vector (the GP's input)

For candidate `(S, w)`, `S = [i_1,…,i_n]`, `w = [w_1,…,w_n]`, and any
per-baseline score map `v` from stages A–D:
```
weighted_mean(v; S, w) = Σ_k w_k · v.get(i_k, 0.5)  /  Σ_k w_k
```
(`features.py:41-43` — missing/unscored baselines fall back to the neutral
`0.5`, same constant as Stage C's clip range midpoint)
```
features(S, w) = [ weighted_mean(heavyweight_score),
                    weighted_mean(activity_semantic_score),
                    weighted_mean(placement_semantic_score),
                    weighted_mean(empirical_baseline_score) ]  ∈ ℝ⁴
```
(`features.py:31-36,79-126`) — deliberately just these four means; no cost
term, no portfolio-shape statistics (task-doc decision, `features.py:26-28`).

---

## 3. Candidate generation — how `(S, w)` itself is produced

Two disjoint generation mechanisms, split by iteration index, both defined
by `config.ITER_LLM_END = 30` (1-indexed iterations):

**Iterations 1–29 (LLM phase).** One candidate per iteration from a
forced-tool-use LLM call conditioned on every baseline's frozen facts
(Stage A/B/C scores) plus the running trial log so far (`llm_proposer.py`,
dispatched at `loop.py:136-145`). Not a closed-form rule.

**Iterations 30+ (BO phase), uncapped for the rest of the run.** A
mechanically enumerated, LLM-free pool:
```
Ω = ⋃_{s=1}^{k}  C(B, s)         (B = eligible baseline pool, |B|=n,
                                   k = max_subset_size, default 4)
```
(`candidate_pool.py:60-68`, `DEFAULT_MAX_SUBSET_SIZE = 4`) — guarded before
enumeration by a cheap `math.comb` bound
(`_estimate_portfolio_count`/`MAX_POOL_PORTFOLIOS = 1e8`,
`candidate_pool.py:135-176`; a real 2026-08-25 incident: an uncapped
44-baseline pool is `2^44 ≈ 1.8×10^13` subsets and OOM-killed the host).

Per subset `S` with `|S| ≥ 2`, up to `n_weight_draws` (default 3)
deduplicated weight vectors are drawn from a **flat Dirichlet prior**:
```
w ~ Dirichlet(1,…,1)     (uniform over the (n−1)-simplex)
```
rounded to 4 dp, renormalized, and kept only if its L∞ distance from every
already-kept draw for that subset exceeds `MIN_WEIGHT_DIFF = 0.03`
(`candidate_pool.py:83-103`). `|S| = 1` always yields the single weight
`w = [1.0]`. The RNG seed is deterministic per-subset
(`md5(f"{seed}:{subset_join}")`, `candidate_pool.py:75-80`), so the pool is
reproducible across runs given the same eligible-baseline list.

---

## 4. The reward — the scalar every trial is judged by

Every real trial, in **either** phase, is scored on the target dataset's
own classes only (Stage 6 below explains why) and reduced to one number:

```
reward(mean_f1, std_f1; κ=0.5)             = mean_f1 − κ·std_f1
```
(`surrogate.py:29-34` — the single-signal macro-F1 building block; no cost
term is subtracted, by explicit design choice, not a zeroed-out placeholder)

```
reward_per_class(pcf1; κ=0.5, "worst_class") = min_c( pcf1[c] ) − κ·std(pcf1_std)
```
(`surrogate.py:37-57` — `min` over classes, not `mean`: `mean(per_class_f1)`
is *by definition* the same number as macro-F1, so blending macro-F1 with
the mean of per-class F1 would be degenerate; `min` is the one per-class
quantity that actually diverges)

```
reward_combined(macro_f1, per_class_f1; α=0.5, κ=0.5)
    = α·macro_f1  +  (1 − α)·min_c(per_class_f1[c])  −  κ·std_f1
```
(`surrogate.py:60-74`; `α = config.REWARD_MACRO_WEIGHT = 0.5`, an untuned
equal-weight default) — **this is the one scalar logged for every iteration
in both phases** (`loop.py:204-218`), so a badly-served minority class
(hidden by an average) can still drag a candidate's reward down.

`macro_f1` and every `per_class_f1[c]` are computed via `harbench`'s own
`evaluate()`, restricted to the target dataset's own label ids only —
`labels=eval_label_ids` passed into `sklearn`-style macro/per-class F1
(`finetune.py:339-357,403-417`; `eval_label_ids` resolved by
`_resolve_target_label_ids`, `finetune.py:954-964`). This restriction is
itself a fix: scoring against the full pooled taxonomy previously diluted
`macro_f1` by *how many baselines got pooled*, not by target performance
(commit `19b05f0`, `444c92f`).

---

## 5. Selection decision I — which candidate runs *next* (acquisition)

BO-phase only. Fit a Gaussian Process on every previously-tried candidate's
`(features, reward_combined)` pair:

```
g ~ GaussianProcessRegressor( kernel = RBF(ℓ=1, fixed) + WhiteKernel(1e-3),
                               normalize_y=True, n_restarts_optimizer=0 )
```
fit on `StandardScaler`-normalized feature vectors `X` (or zero-centered if
only one history point exists) (`surrogate.py:83-95`). Deliberately
**untuned** — the docstring calls this "a proof the mechanism works... not a
statistically tuned model at the trial counts this repo runs."

For every untried candidate `c` in the pool, predict `(μ_c, σ_c) =
g.predict(x_c, return_std=True)` and score by **Upper Confidence Bound**:
```
UCB(c) = μ_c + κ_UCB · σ_c ,    κ_UCB = KAPPA_UCB = 1.0
chosen = argmax_c  UCB(c)
```
(`surrogate.py:26,98-124`). Cold start (empty history) instead ranks by
`mean_heavyweight_score` alone (`bootstrap_rank`, `surrogate.py:77-80`) — in
practice the LLM phase always populates history before BO ever starts, so
this path is a defined fallback, not the normal entry point.

`mean_empirical_baseline_score` (Stage D) is **recomputed every BO
iteration** from the growing `trial_log`, not cached at pool-build time — a
baseline that turns out toxic at iteration 33 measurably discourages
re-picking it at iteration 40 (`loop.py:152-158`).

---

## 6. Selection decision II — which trial is *the winner*

Across **both phases pooled together** (an early LLM-phase trial can beat
every later BO-phase trial):
```
winner = argmax_{t ∈ trial_log}  reward_combined(t)
```
(`loop.py:224-229`, dict key kept as `"best_by_macro_f1"` for an external
JSON contract even though the ranking criterion is the full blended reward,
not raw macro-F1 alone — read this key name as legacy, not literal).

This is the number `optimal_subset_selection` hands to `harbench` as "the
candidate worth a second look" — but it is still a **search-time** number,
computed on the search-view (held-out subjects physically absent from
disk). It is explicitly *not* the reportable result
(`finetune.py:1319-1321`: `pooled_search_time_macro_f1` is "kept for
reference only"). Section 8 covers the confirmatory step that produces the
number that *is* reportable.

---

## 7. User-role assignment — the equations and rules

Every reported F1/accuracy number is only as trustworthy as the user split
that produced it. Three independent, fully deterministic splitting rules
apply, selected by run mode — **no rule ever draws a random train/test split
per run**; every split is either a fixed lookup table or a seeded
reproducible draw.

### 7a. Single-dataset 4-fold CV (`run_finetune`, default mode)

```python
FOLDS = [
    {"test": [1, 2], "val": [3, 4]},
    {"test": [3, 4], "val": [5, 6]},
    {"test": [5, 6], "val": [7, 8]},
    {"test": [7, 8], "val": [1, 2]},
]
```
(`finetune.py:159-164`, `SEED = 42` at `finetune.py:157` seeds everything
*else* — model init, sampler shuffling — not this table, which is fixed)

For fold `k`: `test = FOLDS[k].test`, `val = FOLDS[k].val`,
`train = ~(test_mask ∪ val_mask)` — i.e. **train is defined as the
complement**, computed once in the shared loader
(`train_mask = ~(test_mask | val_mask)`, `src/data/dataloader.py:441-443`),
not enumerated separately. Every subject id `1..8` serves as `test` in
exactly one fold and `val` in exactly one fold (rotation over 8 ids, 2 per
role per fold). **Any subject id beyond 8 that exists in the raw data is in
`train` in every single fold — it is never tested, in any fold** — an
explicit, verified-cross-repo caveat: `held_out.py`'s own docstring calls
this out ("harbench's FOLDS only ever rotates subjects 1-8 through test/val
roles; every dataset's subjects beyond #8 are already used for TRAINING in
every fold", `held_out.py:5-7`).

### 7b. Pooled multi-baseline training (`run_finetune_pooled`, and the pooled
half of `run_finetune_multi_candidate`)

`_pooled_train_val_test_masks` (`finetune.py:911-951`, shared by both call
sites so this rule has exactly one implementation). Per pooled dataset `d`
with raw ids sorted ascending `r_1 < r_2 < … < r_m`:

```
if m ≥ 4:   test(d) = {r_1, r_2}      val(d) = {r_3, r_4}      train(d) = {r_5,…,r_m}
if m < 4:   per-user split impossible → random per-window fallback (below)
```
This is **position-based, not literal-id-based** — the deliberate fix over
an earlier bug where `FOLDS[0]`'s literal ids `[1,2]`/`[3,4]` silently
produced an *empty* val/test split for any dataset not numbered `1..8` (e.g.
`har70plus` uses raw ids `501..518`); see commit `d5e8b2f` and the inline
rationale at `finetune.py:1013-1025`.

**Window-level fallback**, only for a pooled dataset with `< 4` distinct
users (`finetune.py:935-944`):
```
perm = RandomState(SEED=42).permutation(all_window_indices_of_d)
n_test = max(1, round(N · 0.2))     n_val = max(1, round(N · 0.2))
test(d) = perm[:n_test]   val(d) = perm[n_test : n_test+n_val]   train(d) = rest
```
(`WINDOW_SPLIT_TEST_FRAC = WINDOW_SPLIT_VAL_FRAC = 0.2`, `finetune.py:169-170`)
— this is **not subject-independent** for dataset `d` (windows from the same
subject can land in both train and test), an explicit, named limitation
(`finetune.py:1022-1025`: "there aren't enough real subjects to make it so").
It applies **per affected dataset only** — every other pooled dataset in the
same manifest still gets the subject-independent 2/2/rest split above.

### 7c. Permanent held-out subjects (repo: `optimal_subset_selection`,
consumed by `harbench` via `--custom_test_users`/`--custom_val_users`)

```
FOLDS_SUBJECT_IDS = {1, 2, 3, 4, 5, 6, 7, 8}      (held_out.py:33, mirrors 7a's domain)

held_out(T) = the 2 lowest-id subjects of T with raw id ∉ FOLDS_SUBJECT_IDS
val(T)      = the next 2 lowest-id subjects of T with
                  raw id ∉ FOLDS_SUBJECT_IDS ∪ held_out(T)
```
(`pick_held_out_subjects`/`pick_val_subjects`, `held_out.py:36-60` —
deterministic, "the lowest-id ones among the spare pool, for
reproducibility", not random)

```
search_view(T) = symlink view of T's on-disk USER folders
                 with held_out(T)'s folders excluded
```
(`build_search_view`, `held_out.py:63-68`) — **every** search-time trial
(every LLM- or BO-phase candidate, `manifest_builder.build_manifest`'s
`target_entry`) is pointed at `search_view(T)`, so `held_out(T)` is
*structurally* absent from training data for the entire search, not merely
excluded by an id filter that a bug could bypass.

**Final honesty check** — run exactly once, after the search picks a
winner, on the **full, unfiltered** data root:
```
harbench finetune.py --single_split \
    --dataset T --data_root <FULL root, not search_view> \
    --weights <winner's finetuned backbone> \
    --custom_test_users {held_out(T)} --custom_val_users {val(T)}
```
(`held_out.py:86-124`, `finetune.py:1216-1217,1226-1241` for the
`run_finetune_multi_candidate` variant that re-checks winner + companions in
one process instead of `N` subprocess calls). **This is the only evaluation
in the whole pipeline that never influenced a single search decision** — it
is the number that belongs in a paper's results table, not
`pooled_search_time_macro_f1` (Section 6) and not the LLM/BO reward.

`--custom_test_users`/`--custom_val_users` (`finetune.py:1828-1834`)
literally override `FOLDS[0]`'s ids for single-split mode — validated to
require `--single_split` (or `--candidates_manifest_json`) and to always be
given as a pair (`finetune.py:1862-1881`).

### 7d. Mixture-weight realization is not a user-role split, but changes who is
physically present in `train`

`weighted_pool.build_weighted_view` (repo: `optimal_subset_selection`)
realizes a candidate's non-uniform `w_i` by **dropping whole user folders**
(never partial-user rows), since `harbench`'s pooled loader has no
weighting concept of its own — the only way to make an arbitrary `w` take
effect is to make the data on disk already reflect it before `harbench`
loads it:
```
scale_factor = min_i( natural_i / w_i )        (largest pool achievable by pure down-sampling)
target_i     = scale_factor · w_i
```
(`compute_target_counts`, `weighted_pool.py:60-66`) then a greedy,
folder-name-sorted selection keeps users until the running total is closest
to `target_i`, **never below `MIN_USERS = 4`** kept users
(`select_users_for_target_count`, `weighted_pool.py:69-92,33`) — that floor
exists precisely so 7b's `≥4-users` branch still applies after down-sampling
rather than silently tipping a baseline into the window-fallback branch.
A small `w_i` therefore shrinks baseline `i`'s **train** contribution only;
`i`'s own test/val users are still chosen by 7b's position-based rule on
whichever users remain in the view.

---

## 8. Control flow — `run_finetune_multi_candidate` (`finetune.py:1144-1340`)

The one `harbench`-native function that ties Sections 4–7 together for a
single target, replacing `optimal_subset_selection/candidate_ablation.py`'s
older `2·(N+1)`-subprocess orchestration with one process:

<div class="step">

**1. Load target data once** (`finetune.py:1212-1217`) — `load_dataset`
reads the target's **full, non-search-view** root (held-out subjects must
be physically present so `--custom_test_users` can exclude *and* later test
on them). Reused for every held-out-eval call below.

**2. `prior_backbone` baseline** (`finetune.py:1219-1241,1302-1306`) —
`_held_out_eval` on the un-finetuned pretrained backbone: plain CE
single-split fine-tune on `train = all \ (held_out ∪ val)`, test only on
`held_out`. This is the pre-search reference point.

**3. Per candidate `cid` in the manifest list** (`finetune.py:1243-1300,
1308-1325`):
   - `_train_pooled_candidate`: rebuild the **same** CE+SCL pooled trial
     Section 4–6 already ran during search (`set_seed` re-called before
     every stage — one process now runs what used to be separate
     subprocesses, so RNG state must be reset explicitly or results become
     run-order-dependent), using the 7b split, logging
     `pooled_search_time_macro_f1` for reference only.
   - `_held_out_eval` on the resulting finetuned backbone: the 7c honesty
     check, on the SAME held-out subjects as step 2, for a fair
     apples-to-apples comparison.

**4. Write `results.json`** (`finetune.py:1327-1340`) — one row per
candidate: `{candidate_id, baselines: dict(zip(S, w)), test_f1, test_acc,
pooled_search_time_macro_f1, ...}`. This is exactly the file the user's IDE
had open (`.../finetune_multi_candidate/.../resnet/results.json`).

</div>

**Determinism note** (`finetune.py:1169-1176`): every stage explicitly
calls `set_seed(args.seed)` and does `del model` + `torch.cuda.empty_cache()`
— a fresh subprocess used to get a fresh seed/fresh GPU memory for free;
running `2·(N+1)` models sequentially in one process makes both of those an
explicit responsibility instead.

---

## 9. Input / output contracts

| Artifact | Produced by | Consumed by |
|---|---|---|
| `passed_lightweight_dataset.json` | `lightweight-score-unify` | `optimal_subset_selection/lightweight_pool.py` (eligible pool) |
| `<target>_heavyweight_rank.md` | `heavyweight-similar-score` | `optimal_subset_selection/heavyweight_scores.py` |
| dataset cards (`dataset_cards/cards/*.json`) | authored/bootstrapped in `optimal_subset_selection` | `semantic_scorer.py` |
| `--candidates_manifest_json` (`[{candidate_id, S, w, manifest}]`) | `optimal_subset_selection` (post-search selection of winner+companions) | `finetune.py::run_finetune_multi_candidate` |
| `--baseline_manifest` (`[{dataset, sensors, data_root, label_map}]`) | `manifest_builder.build_manifest` | `finetune.py::run_finetune_pooled` |
| `log/loop_state/<target>__<time>.json` | `loop.run_loop` (every iteration) | not read by `harbench`; audit trail only |
| `results.json` (`finetune_multi_candidate` mode) | `finetune.py::run_finetune_multi_candidate` | a human / the paper's results table; not re-consumed by any pipeline stage |

---

## 10. Assumptions baked into the code (and where they'd break)

- **7a's `FOLDS` assumes raw subject ids `1..8` exist and mean "the
  canonical rotation set".** Any dataset numbered differently either
  silently ignores subjects `>8` for testing forever (7a) or would have
  produced an *empty* split before the 7b fix (now guarded — see
  `d5e8b2f` and `create_dataloaders`'s loud `ValueError` at
  `src/data/dataloader.py:455-466`).
- **7c's held-out/val picks assume every target dataset has `≥4` spare
  subjects outside `FOLDS_SUBJECT_IDS`.** `held_out.py:41-46,55-60` raises
  `ValueError` rather than silently degrading if not — there is no
  window-level fallback for the held-out reservation the way 7b has one for
  pooled training.
- **`reward_combined`'s `α=0.5` and `κ=0.5` are untuned defaults**, not
  fit to any validation criterion — stated explicitly in `config.py:70-72`
  and `surrogate.py`'s reward docstrings. A methodology section should cite
  these as fixed hyperparameters of the search, not learned values.
- **The GP kernel is deliberately fixed/untuned** (`length_scale_bounds=
  "fixed"`, `n_restarts_optimizer=0`) — `surrogate.py:1-8` calls this "a
  proof the mechanism works... not a statistically meaningful surrogate at
  the trial counts this repo runs." Cite this if reviewers ask why the GP
  isn't hyperparameter-tuned.
- **`weighted_pool`'s down-sampling never goes below `MIN_USERS=4`
  survivors**, so an achieved mixture weight can be off from the requested
  one for a baseline whose natural size is small — `manifest_builder`
  reports both `requested` and `achieved` weight rather than assuming they
  match (`weighted_pool.py:17-22,123,133-134,138-146`).
- **`pooled_search_time_macro_f1` is computed on the search-view (held-out
  excluded, but NOT independent of the acquisition process that chose this
  candidate)** — it is explicitly flagged non-reportable
  (`finetune.py:1319-1321`); only Section 7c's number is a valid honesty
  check.

---

## 11. How to verify this deck's claims yourself

No pytest-based test in `harbench` exercises the cross-repo selection math
directly (that lives in `optimal_subset_selection/tests/`, run via
`python tests/run_all.py`, mocking the real LLM/subprocess calls). To sanity
check `harbench`'s own splitting code in isolation:

```bash
python -c "
from src.data.dataloader import load_dataset, create_dataloaders
X, Y, U = load_dataset('dsads', ['ACC'])
tl, vl, tel = create_dataloaders(X, Y, U, [1,2], [3,4])
print(len(tl.dataset), len(vl.dataset), len(tel.dataset))"
```
and confirm the three counts partition `len(U)` with no overlap (`train_mask
= ~(test_mask | val_mask)` guarantees this algebraically, but a live run
also catches a silently-empty split via the `ValueError`s at
`src/data/dataloader.py:455-466`).

For the full multi-candidate flow end to end, a real (if slow) integration
check is running `run_finetune_multi_candidate` with a tiny
`--candidates_manifest_json` (one candidate) and `--epochs 1` and confirming
`results.json` has exactly `1 + len(candidates)` rows with distinct
`weights_path`s.

---

## 12. How to maintain this

- **To change the reward function**: edit `surrogate.py`'s `reward_combined`
  only — `loop.py` and `finetune.py` never hardcode the formula, they call
  it. Update `config.REWARD_MACRO_WEIGHT` for the blend weight, not a
  literal in `surrogate.py`.
- **To change how many held-out/val subjects are reserved**: `held_out.py`'s
  `reserve_target(n_held_out=2, n_val=2)` — but `harbench`'s own 7a/7b rules
  (`FOLDS`, `_pooled_train_val_test_masks`) independently assume exactly 2
  test + 2 val per fold/dataset; changing one without the other desyncs
  which subjects are "spare" for 7c's reservation.
- **To change the BO candidate pool's size**: `candidate_pool.py`'s
  `max_subset_size`/`n_weight_draws` — always re-check
  `_estimate_portfolio_count` against `MAX_POOL_PORTFOLIOS` before raising
  either; the 2026-08-25 OOM incident is the reason that guard exists.
  `harbench` itself has no cap of its own — it trusts whatever manifest it's
  handed.
- **To add a new held-out/val selection rule** (e.g. stratify by class
  balance instead of lowest-id): only `held_out.py`'s `pick_held_out_
  subjects`/`pick_val_subjects` need to change; `harbench`'s
  `--custom_test_users`/`--custom_val_users` contract is agnostic to how the
  ids were chosen.
- **To change the pooled test/val split (7b)**: touch
  `_pooled_train_val_test_masks` (`finetune.py:911-951`) only — it's shared
  by `run_finetune_pooled` and `run_finetune_multi_candidate` specifically
  so this rule has exactly one implementation; do not fork it per call site.

---

## Appendix — methodology-section-ready summary

> For each target dataset, baseline datasets are first screened for signal
> quality (a per-dataset score in `[0,1]`, gated at `mean − 3σ` over the
> historical score population) and then re-ranked against the target by a
> leave-one-out z-scored blend of embedding silhouette and MMD. A search
> procedure — 29 LLM-proposed iterations followed by Gaussian-process/UCB
> optimization over a mechanically enumerated subset-and-Dirichlet-weight
> pool (subset size ≤ 4, ≤ 3 weight draws per subset) — evaluates each
> candidate `(S, w)` with a real cross-entropy + supervised-contrastive
> finetuning trial, scored only on the target dataset's own classes. Every
> trial is reduced to `reward = 0.5·macro-F1 + 0.5·worst-class-F1`, and the
> maximum-reward trial across both search phases is selected as the winning
> recipe. Two subjects per target dataset (the lowest-numbered subjects
> outside the standard 4-fold rotation's `{1,…,8}`) are permanently
> reserved and physically excluded from every search-time dataset view;
> after the search concludes, the winning recipe's finetuned backbone is
> re-evaluated exactly once against these held-out subjects, and this
> number — never seen during search — is reported.

Adjust the constants (`α=0.5`, `κ=0.5`, `κ_UCB=1.0`, 29/30 iteration split,
2 held-out + 2 val subjects, subset cap of 4) to whatever values a given
run actually used — every one of them is a named constant in `config.py` /
`surrogate.py`, not hardcoded inline, so a specific run's exact values are
recoverable from its logged config.

---

# Questions?

`harbench/finetune.py` (this repo) · `optimal_subset_selection/{loop,
surrogate,features,candidate_pool,held_out,weighted_pool,manifest_builder,
trial_runner}.py` · `heavyweight-similar-score/scoring.py` ·
`lightweight-score-unify/lightweight_check.py`
