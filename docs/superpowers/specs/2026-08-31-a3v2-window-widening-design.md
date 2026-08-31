# A3 v2 — widened slice-sampling window, design

**Status:** revised after a first Opus review found 3 Critical + 8
Important + 7 Minor findings (all addressed below). User's explicit
call: one review round is enough this time, skip the second
confirmation pass — approved, ready for the implementation plan.
**Plan item:** the "`RSNA_WINDOW` widening" lead, deferred at A2 v2's
own §8 ("Open items carried forward") and reinforced by a 2026-08-31
forum-mining pass — see [[project-rsna-phase-status]].
**Depends on:** the vendored reference kernel
(`data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`,
`RESOURCES.md`'s `stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache`
entry), A2 v2's architecture (`src/model.py`, `src/dataset.py`,
`src/features.py` — unchanged, this reuses them), A2 v2's fold-0 real
result (`notebooks/09v1_a2v2_multigroup_baseline.ipynb`, 0.7956).

## 1. Overview

A2 v2's pooled 4-fold result (0.8009) improved the weak-finding cluster
by recombining the 3 anchor groups A3's cache already extracts — but
those 3 groups all sit inside a **narrow 35-65% window** of each
series' ordered slice stack (confirmed against the reference kernel's
own `RSNA_WINDOW=0.35,0.65` pin). A2 v2's own spec (§1, §8) flagged this
explicitly as a distinct, un-tested hypothesis: recombining anchors
*inside* the narrow window is not the same experiment as *widening* the
window to reach the true periphery, where `dreaddevelopment`'s public
notebook (`RESOURCES.md`) attributes a real, measured gain specifically
on `mcl_injury`/`lateral_meniscus_tear` to sampling 6-94% of the stack
instead of a centre-only window.

**2026-08-31 forum-mining update** (see [[project-rsna-phase-status]]):
`dreaddevelopment` posted a real, updated number for the same
architecture — **0.936-0.938 public LB** (up from the 0.924/0.9167-gold
figure already cited) — plus a real ablation (94 vs. 42 inference
windows/study, only ~0.006-0.008 LB apart). Read plainly, a ~0.007 gap
for a 2.2x coverage cut is closer to a *flat* coverage-vs-score curve in
their own stronger pipeline than strong reinforcement — real evidence,
but pointing at a small expected effect, not a large one (folded into
§4's decision rule below, per the first Opus review's I4 finding).
Separately, a closer reading of our own vendored reference kernel (not
something examined at this depth during A3) found a **wider per-plane
window default already present in its `read_slot()` code**, unused by
our current cache because a separate setup cell overrides it:

```python
PLANE_WINDOW = {"Sagittal": (0.10, 0.90), "Axial": (0.10, 0.90),
                "Coronal": (0.15, 0.85)}
```

explicitly attributed in the source's own comment to `wguesdon`'s
reasoning ("the menisci sit at the medial and lateral extremes of a
sagittal stack ... a coronal stack's useful anterior-posterior range is
narrower"). Our existing cache never used this — a setup cell in the
same notebook unconditionally sets `os.environ["RSNA_WINDOW"] =
"0.35,0.65"`, which `read_slot()` checks *after* `PLANE_WINDOW` and
overrides with when present.

**Important correction, found by the first Opus review (C3): we have
never executed this build script ourselves.** `notebooks/04v2_slot_cache_integration.ipynb`
points `CACHE_DIR` at `stevenleehans`'s own published Kaggle Dataset
Output, re-uploaded by the user — A3 integrated an already-built
artifact, it did not run `build_pixel_cache.py`. This matters here
specifically because `PLANE_WINDOW`'s own comment says its values
*replaced a 20-80% window* as "part of the ordering fix" — i.e. this
file's slice-selection/ordering code has a documented history of
changing. §2.1 below adds a real check for this before trusting the
comparison: a full field-by-field diff of the current cache's
`cache_meta.json` fingerprint against the new build's, not just the
`window` field.

**Decision (user, 2026-08-31, choosing between two sourced options with
no forum evidence to break the tie further — see the chat record):**
adopt the kernel's own `PLANE_WINDOW` defaults (stop overriding
`RSNA_WINDOW`) rather than inventing a uniform window number from a
different, confounded pipeline (`dreaddevelopment`'s own gain mixes
window width with an entirely different architecture — CoAtNet, 5
slots, no isolated ablation of window width alone). This spec's own
choice is the one already validated inside the exact pipeline our
current cache depends on.

**Second decision (user, 2026-08-31): stage this separately from the
224px→336px resolution fix**, not bundle them. Real, previously-unknown
cost asymmetry found while reading the source's `CFG` class: window
width costs **nothing extra** (same slice count decoded, same cache
shape/size, ~11 GiB); resolution costs a real, quantified **2.25x**
compute/storage increase (the source's own comment: `130mm @ 224px =
8.13 mm/token, 130mm @ 336px = 5.42 mm/token (2.25x compute)`).
Bundling them would also confound which one drove any result, the same
class of mistake `RSNA_WINDOW` itself was invented to avoid (the
source's own `exp-016` note: "3 central slices beat 9 spread slices" was
confounded between slice *count* and slice *centrality* until the two
were separated). **This spec covers window-only, at the existing
224px/130mm crop.** The resolution fix stays a separate, later,
conditional decision (§8).

**This spec's scope, precisely:** rebuild the A3 slot cache using the
same vendored pipeline, same `crop_mm=130`, same `n_group_max=3` (9
slices/slot, 3 anchor groups), same 6 named slots — the *only* thing
that changes is which slice indices `read_slot()` selects within each
series. Because the cache's output shape is unchanged
(`(n_studies, 6, 9, 224, 224)` uint8, same as today), **no downstream
code changes are needed**: `select_group()`, `expand_slot_groups()`,
`SlotCacheDataset`, `SlotAttentionModel` all operate on shape alone and
have no dependency on which physical slice positions the cache's pixels
came from. Confirmed by re-reading `src/features.py`/`src/dataset.py`/
`src/model.py` directly — none references `RSNA_WINDOW` or any
window-derived quantity.

## 2. Cache rebuild

### 2.1 Reuse, not rebuild (same principle as A3 itself) — but a real first run, not a re-run

New notebook, CPU-only, `notebooks/13v1_a3v2_window_cache_build.ipynb`.
Runs the vendored `build_pixel_cache.py` script (the reference kernel's
own cell) with:

- `RSNA_CROP_MM=130` (unchanged — isolates window as the only changed
  variable, per §1's confound-avoidance reasoning)
- `RSNA_N_GROUP_MAX=3` (unchanged)
- `RSNA_IMG` **not set** (defaults to 224 — unchanged; 336px is
  explicitly out of scope, §8)
- `RSNA_WINDOW` **not set** — this is the entire intended change.
  `read_slot()` falls through to its own `PLANE_WINDOW` per-plane
  defaults.

**This is this project's first-ever execution of `build_pixel_cache.py`**
(§1's correction) — not a re-run of the pipeline that produced our
current cache, which was `stevenleehans`'s own separate build. Two
consequences, both required before spending real time on the full
corpus:

**Step 0 — fingerprint diff, before the probe.** `main()` computes
`fp = fingerprint(n_group)` immediately after slot-picking, before the
`--probe` early-return — so this is available within minutes, no decode
needed. Print it, and diff it field-by-field against the *current*
cache's own `cache_meta.json` (still attached as a Dataset input):
`img`, `crop_mm`, `pad_short_fov`, `lat_from_geometry`, `group`,
`n_group`, `slots`, `seed` — every field except `window` (the one
field expected to differ). `04v2` only ever asserted `crop_mm`/`group`/
`n_group`/`slots`/`img` when integrating the current cache — `pad_short_fov`,
`lat_from_geometry`, and `seed` were never checked, so whether the
current cache's `cache_meta.json` even contains matching values for
those is genuinely unknown until this diff runs. Any unexplained
difference beyond `window` is a real confound to resolve (or explicitly
accept and document) before §4's comparison means anything — if
`pad_short_fov`/`lat_from_geometry` are absent from the current cache's
meta entirely, that's real evidence the published build predates
fields this version of the file tracks, strengthening the case that
more than just the window may differ between the two caches.

**Step 1 — pre-flight, before committing to the full ~500GB decode:**
run the script's own `--probe N` flag (decodes N studies, projects the
full-corpus time — the script's own `estimate_decode_time()`).
**Acceptance criterion, corrected per the first review (I1/I2):** the
kernel's own markdown records a documented probe bias — a 40-study probe
previously measured **9.0 slot-series/s** and projected 0.71h, while the
real sustained rate was **6.25/s** taking 1.03h (the probe ran 44%
optimistic). Comparing this run's *projected total time* against the
"~55 min" figure in the script's docstring is therefore not the right
check — that figure is itself a GPU-adjacent estimate the same notebook
corrects to **~64.2 min measured, CPU-only**. The right check is
**rate-to-rate**: run `--probe 40` (matching the documented sample size)
and compare the measured slot-series/s directly against the documented
9.0/s baseline, not against a derived wall-clock target. A large,
unexplained deviation either direction is worth understanding before
the full run, not just a slower-than-hoped number.

### 2.2 Fingerprint and shard layout

The script's own `fingerprint()` includes `window` (read from
`os.environ.get("RSNA_WINDOW", "default")`) specifically so a cache
built without the override is distinguishable from the current one —
confirmed this is already handled, not something to add. Output layout
identical to the current cache (train shards + test shards, `cache_meta.json`)
— same `--shards 4` convention already used, since neither shard count
nor per-shard size changes. Upload as a new, separate Kaggle Dataset
(not overwriting the current cache — every downstream notebook that
still depends on the narrow-window cache, including any future A2 v2
resubmission, must keep working unaffected).

**Completeness check, qualified (M5):** the source's own docstring notes
an empty shard (e.g. `test`'s 3 placeholder studies over 4 shards
producing one empty block) logs as "empty shard (no studies in this
block)" and is recorded as complete, by design — not an `INCOMPLETE`
warning. Treat that specific message as expected, not a failure; only a
real `INCOMPLETE, still missing: ...` line (from a truncated shard) is
the actual failure signal.

## 3. Training pilot (fold 0 only)

New notebook, `notebooks/14v1_a2v3_window_only_fold0_pilot.ipynb`,
mirroring `09v1`'s own structure closely (self-contained, no `import
src`, same hand-kept-copy convention). Points `CACHE_DIR` at the new
Dataset from §2; everything else — `SlotCacheDataset(expand_groups=True)`
(18 pseudo-slots, A2 v2's winning architecture), `SlotAttentionModel`,
`masked_finding_attention` — copied verbatim, no logic changes, per §1.

**Cache-identity guard (I5, new):** before training, assert the
attached cache's own `cache_meta.json` matches what this notebook
expects — `window == "default"`, plus `crop_mm`/`img`/`group`/`n_group`/
`slots` against their known values — same pattern `04v2` already
established for the current cache. Two near-identically-named Datasets
will exist after §2; a stale `CACHE_DIR` would silently rerun A2 v2
under a different name instead of testing anything new.

**Fold-assignment re-assertion (I6, restored):** same as A2 v2's own
spec's step 1 — regenerate the fold split from `report_group_key`/
`scanner_fingerprint` grouping (unrelated to the cache rebuild, but
worth re-asserting since this notebook is a fresh execution) and assert
fold 0's val set matches the recorded 1,307 studies / 17 gold before
trusting any comparison against 0.7956.

**Checkpoint selection — report both readouts from one run, not SWA
alone (C1/C2, corrected).** The first Opus review found two real
problems with reporting only an SWA number:

1. **0.7956 (the baseline) was itself selected as the best of 12
   epochs on the same noisy gold gate this spec is trying to avoid
   over-trusting.** Comparing that selection-inflated number against an
   unselected SWA average biases the gate against the candidate before
   any real effect is measured — the reference kernel's own comment
   puts a real number on this: two identical-config, identical-seed
   runs that differed only in which epoch got restored (12 vs. 5)
   landed **0.0120 apart on pooled OOF**, and a single 17-gold-study
   fold is noisier still.
2. **The reference kernel's own SWA branch never computes gold AUC for
   the averaged model** — its `sc = macro_auc(yv, pv)` line scores
   against `yv = (Y[va] > 0.5)`, the *derived* (weak) labels, and its
   own log line says so explicitly (`"averaged last N epochs, NO
   selection (derived {sc:.4f}; ...)"`). There is no upstream mechanism
   to copy for a gold-AUC SWA readout — `14v1` must add one itself (the
   same `predict()`-then-`macro_roc_auc()` pattern already used
   per-epoch, just called once more after loading the averaged weights
   onto the fold-0 gold subset).

**Fix, both issues, one run:** train with the existing best-epoch
tracking (`if gold_auc > best_gold_auc: ...`, matching every prior
notebook in this project) **and** accumulate the SWA average in
parallel (both mechanisms coexist in the reference kernel's own
`train_fold`, they are not mutually exclusive). At the end, evaluate
**both** the best-epoch checkpoint and the SWA-averaged checkpoint on
fold 0's gold subset. §4's gate compares the **best-epoch** number
against 0.7956 (apples-to-apples with how 0.7956 was itself produced).
Report the **SWA** number alongside it as the forward-looking figure,
plus the gap between the two — this project's first real, on-our-own-
data measurement of its own selection-noise premium, directly useful
context for [[feedback-checkpoint-selection-noise]] even independent of
whether the window change itself helps.

**SWA epoch count: `swa_epochs=3`, with real justification (M2,
corrected from a hand-wave).** Averaging the last 3 of 12 epochs, under
`09v1`'s own `OneCycleLR` (no `pct_start` passed → PyTorch's default
0.3, LR peaks at epoch ~3.6): epochs 10/11/12 sit at roughly
28.3%/13.3%/3.5% of `max_lr`, monotonically annealing — well past the
schedule's peak, no warmup contamination. (Under the reference kernel's
own `pct_start=0.15` convention the same 3 epochs would be
19.9%/9.2%/2.4% — same qualitative shape either way.) The reference
kernel exposes the mechanism (`RSNA_SWA_EPOCHS`) but states no chosen
value anywhere in its own file (confirmed by searching all 26 cells) —
`swa_epochs=3` is this project's own choice, justified by the schedule
math above, not copied from a sourced number.

**Corruption hazard, must be ported deliberately (I7):** the reference
kernel's own comment on its SWA-accumulation line is explicit:
`.clone()` is load-bearing — `.cpu()`/`.float()` are no-ops on a tensor
already CPU float32, so omitting `.clone()` makes `swa_state` alias the
live model parameters, and the in-place `+=` used to accumulate the
running average then corrupts the model mid-training. The bug "survives
on GPU only because `.cpu()` happens to copy there" — i.e. it can be
silently absent on one device and present on another. `14v1` does not
call the reference kernel's `train_fold` directly (it mirrors `09v1`'s
own loop, with gradient accumulation), so this block must be **hand-
ported with this exact hazard in mind**, not assumed safe by proximity
to working source code.

**Seed, stated as a real limitation, not fixed retroactively (M4):**
searched `09v1` for `torch.manual_seed`/`np.random.seed` — neither is
called anywhere. **0.7956 is therefore a single unseeded draw**, and
`gold_tol=0.03` is being applied against a run-to-run noise floor this
project has never directly measured for this notebook family (the
reference kernel's own SWA comment is explicit that the two-epoch
selection gap it measured *is* that floor, for its own pipeline — ours
is untested). `14v1` should pin a seed (`CFG.seed`-style, matching the
reference kernel's own convention) for its own run, and this spec
states plainly that comparing a seeded run against an unseeded baseline
is an accepted, not resolved, limitation.

**Hyperparameters: otherwise unchanged from A2 v2** (`AdamW`,
`lr_backbone=8e-6`, `lr_head=1e-3`, `weight_decay=0.02`, last 6
transformer blocks unfrozen, 12 epochs, OneCycle,
`MICRO_BATCH=8`/`ACCUMULATE_STEPS=4` — A2 v2's own real measured-safe
VRAM split, expected to still apply since input shape is unchanged from
A2 v2, re-verified by the pre-flight smoke test, not assumed).

## 4. Gate (fold-0 pilot only — matches A2 v2's own §5.1 pattern)

**Baseline:** A2 v2's real fold-0 result
(`notebooks/09v1_a2v2_multigroup_baseline.ipynb`'s real output),
**0.7956, an 11-finding macro** (`oa_lateral_compartment` undefined,
single-class, among fold 0's 17 gold studies — same exclusion as A2 v1's
own fold 0).

**Macro check (pass/fail):** `candidate_gold_macro - 0.7956 >= -0.03`
(`gold_tol=0.03`, same tolerance and same stated caveat as A2 v2's own
§5.1 — legitimate at the macro level, borrowed from a pooled-level
measurement, so a marginal breach in the `-0.03` to `-0.05` range is
"borderline, use judgement," not an automatic hard stop, per A2 v2's
same reasoning).

**Directional read:** `medial_meniscus_tear` (A2 v2 fold-0: 0.597) and
`lateral_meniscus_tear` (A2 v2 fold-0: 0.652) remain the two gating
findings — both have a real, defined fold-0 baseline and reasonable
weak-label quality, same as A2 v2's own spec.

**`mcl_injury`: reported as context, not gating (I3, corrected).**
A2 v2's spec excluded `mcl_injury` for two reasons: (a) its fold-0
baseline had the widest noise band of the cluster, and (b) `08v1`'s real
finding that 2/9 gold `mcl_injury`-positive studies have a completely
blank `COR_T1` slot, capping how much a *group-recombination-only*
change could move it. Reason (a) does not carry forward as originally
worded — A2 v2's own real fold-0 `mcl_injury` result is **0.800**, not
the 0.933 figure A2 v2's spec quoted (that number was A2 v1's fold-0,
carried over in error). Reason (b) also does not fully transfer: the
22% cap is specific to the `COR_T1` slot, but widening the window also
changes the **coronal** sampling range (0.35-0.65 → 0.15-0.85), which
also touches `COR_FLUID_FS` — a different coronal slot the same 2
studies may still have. So this specific change, unlike A2 v2's, is not
provably a no-op for `mcl_injury`. Given the small-n noise concern still
stands on its own, `mcl_injury`'s delta is reported alongside the gate
(and folded into the pre-agreed override rule below) but does not gate
by itself. `oa_lateral_compartment` stays excluded entirely — no fold-0
baseline exists to compare against.

**Decision rule — pre-agreed now, not relitigated after an ambiguous
result (I4, per the user's explicit 2026-08-31 choice).** The first
Opus review flagged a real risk: §1's own cited evidence (a ~0.007 LB
gap for a 2.2x coverage change in a stronger pipeline) points to a small
expected effect, and this exact "both findings must move ≥0.10" rule
already produced an "inconclusive" result once, on A2 v2's own fold-0
pilot, which the user then overrode by hand to scale anyway. Rather than
risk replaying that after another expensive pilot run, the override
condition is fixed here in advance:
- Macro check fails → **stop**, report negative, do not scale.
- Macro check passes **and at least one** of `medial_meniscus_tear` /
  `lateral_meniscus_tear` moves positively by more than ~0.10 (roughly
  one Hanley-McNeil point-estimate SE at this fold's sample size — SE
  values reused from A2 v2's own spec, 0.144/0.141, computed at A2 v1's
  fold-0 AUCs, not recomputed at this baseline; immaterial to where the
  ~0.10 bar falls) — **and** `mcl_injury`'s delta is reported in the
  same write-up — → **scale to the remaining 3 folds.** (Loosened from
  "both must move" to "at least one," decided now rather than after
  seeing the result, specifically to avoid re-litigating a borderline
  outcome the way A2 v2's own pilot had to be.)
- Macro check passes but **neither** finding moves positively by more
  than the noise bar → stop, report as inconclusive (not disproved) —
  same judgement-call framing as A2 v2's spec, now genuinely a distinct,
  stricter branch from the one above.

## 5. Explicitly out of scope for this spec

- **The 224px→336px resolution fix.** A separate, later, conditional
  decision — only worth its real 2.25x compute/storage cost if this
  window-only pilot shows a real signal first (§1). If it does, the
  natural next step is a second cache rebuild combining both (or
  resolution alone, isolated the same way) — not designed here. **Note,
  corrected (M6):** this follow-up needs real `src/` changes when it
  happens, not just a second cache rebuild — `SlotAttentionModel.__init__`
  doesn't forward an `img_size` parameter to `build_backbone()` (pinned
  to its 224 default) and `src/config.py::SLOT_CACHE_IMG_SIZE` is
  hardcoded to 224. Priced correctly here so that later decision isn't
  under-estimated.
- **Porting the window change into the live-decode submission path**
  (needed only if this graduates to a real submission, same as A2 v2's
  own `07v1`→`11v1` port happened only after graduation). **Corrected
  (I8):** the window constant does not live in `src/preprocess.py`
  (checked directly — that file has no window logic at all, only
  header/fat-sat/laterality helpers). It lives inline in
  `notebooks/11v1_a2v2_submission_inference.ipynb` as a single uniform
  `WINDOW = (0.35, 0.65)` tuple. Adopting `PLANE_WINDOW` there means
  converting that into a per-plane lookup keyed off each slot's plane
  string, not swapping two float constants — real, if still deferred,
  work. §1's "window width costs nothing extra" claim is scoped to the
  cache-build step only; this later graduation work is not free.
- **Scaling to the full 4-fold pooled gate** (§4's own decision rule) —
  designed only if/when the fold-0 pilot says to scale, matching A2 v2's
  own two-tier precedent exactly.
- **Adopting SWA for anything other than this new notebook** — A2 v1's
  and A2 v2's existing checkpoints/notebooks are untouched; this is a
  forward-only process change, not a retroactive one
  ([[feedback-checkpoint-selection-noise]]'s own explicit scope).

## 6. Testing strategy

No `src/` changes in this spec (§1 — the whole point is that existing
code needs none), so no new unit tests are owed by this plan. The
existing suite (90/90 passing as of A2 v2's graduation) is not expected
to change. Validation is entirely real-Kaggle-output-based, same
notebook-before-src discipline as every prior item:

1. `13v1`: Step 0's fingerprint diff (§2.1) run and reported before the
   probe; `--probe 40`'s measured slot-series/s compared against the
   documented 9.0/s baseline (§2.1's corrected criterion, not a
   wall-clock comparison); full build must report `complete: N/N
   shards` (an expected "empty shard" message for `test`'s placeholder
   block is not a failure, §2.2); `cache_meta.json`'s `window` field
   must read `"default"` rather than `"0.35,0.65"`.
2. `14v1`: the cache-identity guard and fold-assignment re-assertion
   (§3, I5/I6) before any training; pre-flight smoke test (overfit 8
   real studies, VRAM/host-RAM checks) before spending real GPU time on
   the fold-0 run, same practice as every prior training notebook in
   this project.
3. `14v1`'s own "Real output" section filled in with the real per-finding
   table, **both** the best-epoch and SWA-averaged gold AUC readouts
   (§3), the §4 macro check (on the best-epoch number), the directional
   read including `mcl_injury`'s context delta, and the resulting
   scale-or-stop decision per the pre-agreed rule — computed for real,
   not estimated.

## 7. Process note

Per [[feedback-opus-review-spec-before-implementing]]: this spec gets an
Opus-model review before the implementation plan is written, same
process as A2 v2's own spec (which went through two review rounds and
had real gate-design bugs caught before any code was written).

**First review pass (2026-08-31):** confirmed the spec's central
technical claim true against the real code (`src/features.py`/
`dataset.py`/`model.py` are genuinely shape-only, no window dependency;
`PLANE_WINDOW` genuinely applies per-plane correctly; decode cost is
genuinely window-independent) and every number carried forward from
A2 v2/`08v1` accurate. Found 3 Critical issues — (C1) the proposed SWA
comparison was biased against the candidate by comparing an unselected
average to a selection-inflated baseline; (C2) the described SWA
gold-AUC readout doesn't exist in the source and must be built; (C3) the
spec's "same pipeline, one variable" premise was wrong — this project
has never run `build_pixel_cache.py` before, the current cache is the
external author's own published Output — plus 8 Important and 7 Minor
findings (probe-acceptance miscalibration, a wrong file citation for the
eventual live-decode port, a missing cache-identity guard, a missing
fold re-assertion, an unaddressed SWA-porting corruption hazard, an
`mcl_injury` exclusion argument that didn't fully transfer, and more).
**All addressed above** (§1-§6 revised: dual best-epoch/SWA readout,
gold-AUC-for-SWA added, a fingerprint-diff step added before the probe,
corrected probe/timing criteria, `mcl_injury` demoted to context-only,
a pre-agreed override rule replacing the original three-way branch per
the user's explicit 2026-08-31 choice, cache-identity and fold-assignment
guards restored, the SWA `.clone()` hazard named explicitly, the live-
decode citation corrected, and the 336px follow-up's real `src/` cost
priced in). `RESOURCES.md` updated with the two new forum sources this
spec cites (`starkhushi`'s label-benchmark notebook, `dreaddevelopment`'s
Raptor Weights update).
