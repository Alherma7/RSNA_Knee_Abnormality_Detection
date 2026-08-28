# A2 v2 — multi-group slot attention, design

**Status:** approved by user 2026-08-28; revised after a first Opus
review found real gate-design errors (§5 rewritten), then confirmed
sound by a second Opus review with minor wording fixes applied (§7) —
ready for the implementation plan.
**Plan item:** follow-up to A2 v1, prompted by the 2026-08-28 weak-finding
investigation (see [[project-rsna-phase-status]] and
`notebooks/08v1_meniscus_mcl_slot_group_check.ipynb` /
`08v2_meniscus_mcl_slot_group_check.ipynb`).
**Depends on:** A2 v1 (`src/model.py`, `src/dataset.py`, all unchanged in
architecture terms — this reuses them), A3 (slot cache, unchanged),
`08v1`/`08v2`'s real Kaggle output (motivating evidence).

## 1. Overview

A2 v1's pooled 4-fold CV found 5 findings clustered at 0.61–0.66 gold
macro-AUC, clearly below the other 7 (0.79–0.88):
`mcl_injury` (0.6145), `oa_lateral_compartment` (0.6325), `synovitis`
(0.6559), `lateral_meniscus_tear` (0.6584), `medial_meniscus_tear`
(0.6635). A 2026-08-28 investigation (live Kaggle forum, real public
notebooks, and two purpose-built local diagnostic notebooks run on
Kaggle) narrowed this to two candidate causes:

1. **Resolution** (`wguesdon/rsna-knee-dinov2-at-meniscus-resolution`,
   `RESOURCES.md`) — confirmed real but partial against our own cache
   (`mm_per_px=0.5804`): resolves 2mm+ features, not the smallest ~1mm
   end of the clinical 1-3mm meniscal-tear range. **Out of scope for
   this spec** — fixing it needs a full A3 cache rebuild from raw DICOM
   (expensive), a separate, deferred decision (user's explicit choice,
   2026-08-28).
2. **Centre-anchor-only slice selection** — A3's cache samples 3 anchor
   groups per slot from a *narrow* window (`RSNA_WINDOW=0.35,0.65`), and
   A2 v1 fixes `group_index=1` (centre only). `08v1`'s visual read (7
   real `mcl_injury` rows, 6 real `lateral_meniscus_tear` rows) suggested
   group 1 looks busier/less clean than groups 0/2 for these findings —
   but `08v2`'s `acl_injury` **negative control** showed the same
   busy-centre pattern for a *central*, healthy-cluster finding too,
   weakening (not disproving) the read. Two non-radiologists have pushed
   visual inspection as far as it usefully goes — **the user's explicit
   decision is to settle this empirically with a real training run, not
   more images.**

**Expected effect size — stated up front so a null result gets read
correctly.** The cache's own build source
(`data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`,
`read_slot`) places the 3 anchor groups at exactly **35% / 50% / 65%**
of each series' ordered slice stack — groups 0 and 2 are each only 15
percentage points of stack depth from the centre, and all three stay
inside the central 30%. `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`
(`RESOURCES.md`), the source whose measured claim motivates this whole
experiment, samples **6-94%** of the stack and attributes its gain to
reaching the true outer slices. This experiment does not reach the true
periphery — it only tests whether the 3 anchors *already inside* the
narrow window help when used together versus centre-only. A null result
here is informative about that narrower question; it is weak evidence
that slice position doesn't matter at all, and should not be read as
closing off the wider `RSNA_WINDOW`-widening hypothesis (§8).

**This spec's scope: test hypothesis 2 only.** Replace `group_index=1`
(6 slots, 1 group each, 3 channels/slot) with **all 3 groups on all 6
slots** (18 "pseudo-slots", 3 channels each) — uniform across every
slot rather than special-casing just `COR_T1`/`SAG_FLUID_FS` (user's
explicit choice: more principled, tests whether the narrow-window
problem generalizes, at the cost of ~3x backbone compute).

**Key finding from this session's design walkthrough: `src/model.py`
needs zero code changes.** `masked_finding_attention()`/
`SlotAttentionModel` are already generic over the slot count — `n_slots`
is a constructor parameter, not hardcoded. Verified directly against the
real code (`src/model.py:39-141`) and against the real `timm` DINOv2
backbone (`patch_embed.proj = Conv2d(3, 384, kernel_size=14, stride=14)`,
confirmed 3 input channels expected — this is *why* groups must become
separate pseudo-slots, not concatenated channels; `select_group()`'s own
docstring already documents that `[0,1,2]` concatenates on the channel
axis, which would produce a 9-channel image the pretrained backbone
cannot consume). All the real work is a new data-layer reshape.

## 2. Data flow

### 2.1 New reshape function (`src/features.py`, next to `select_group`)

`expand_slot_groups(cache_slot_stack, slot_mask)` — takes one study's
full slot stack `(6, 9, H, W)` and its `(6,)` slot-presence mask, and
returns:
- `(18, 3, H, W)` pseudo-slot images: for each of the 6 slots, its 3
  groups (`select_group`'s existing group 0/1/2 indexing:
  `cache[..., g*3:(g+1)*3, :, :]`) become 3 independent 3-channel
  entries, not one 9-channel entry.
- `(18,)` pseudo-slot mask: each real slot's single mask bit
  **replicated 3x**, one per group — the cache does not track per-group
  presence separately (confirmed via `cache_meta.json`/`load_slot_cache_shard`,
  A3), only per-slot, so a slot's 3 groups are always jointly present or
  jointly absent.

**Ordering convention, fixed here, not deferred: slot-major.** Pseudo-slot
index `s*3 + g` for real slot `s` (0-5) and group `g` (0-2) — i.e.
`slot0_g0, slot0_g1, slot0_g2, slot1_g0, ..., slot5_g2`. Implementation
must satisfy, for every `s, g`:
`expand_slot_groups(stack, mask)[0][s*3+g] == select_group(stack[s], g)`
and `expand_slot_groups(stack, mask)[1][s*3+g] == mask[s]` — this is
also §6's core unit test.

Calls `select_group()` internally for the per-group extraction (it
already indexes the slice axis correctly) rather than re-implementing
that math — `expand_slot_groups` only reshapes/stacks its outputs onto
the new pseudo-slot axis and replicates the mask.

### 2.2 `SlotCacheDataset` (`src/dataset.py`) — backward-compatible, not a replacement

`SlotCacheDataset.__init__` gains one new parameter,
**`expand_groups: bool = False`**. Default `False` preserves A2 v1's
exact existing behaviour byte-for-byte (`select_group(stack, group_index)`,
6 slots; `group_index` keeps its existing default of `1` — it is
*optional* today, per `src/dataset.py:46`, not required, so the fix here
must not make it required) — this keeps `tests/test_dataset.py`'s
existing shape assertions (`(6, 3, 4, 4)`) and its `group_index=0` vs
`group_index=1` selection tests passing unmodified (two of the five
existing tests construct the dataset without passing `group_index` at
all — an implementer requiring it would break exactly the tests this
flag exists to protect), and keeps A2 v1 reproducible from `src/`
without this change touching it. When `expand_groups=True`, `__getitem__`
calls `expand_slot_groups(stack, slot_mask)` instead (→ `(18, 3, H, W)`
images, `(18,)` mask), and `group_index` is ignored. Validate explicitly:
raise if `expand_groups=True` and `group_index` is passed as anything
other than its default (`1`) — e.g. `expand_groups=True, group_index=0`
must raise, but `expand_groups=True` alone (leaving `group_index` at its
default `1`) must not, since there is no way to distinguish "not passed"
from "passed as 1" once `group_index=1` is the default — that's a
deliberate, accepted blind spot (the ignored value in that one case
happens to equal the default), not a bug. No other change to the
dataset's logic (shard resolution, label lookup, mmap access pattern all
stay exactly as A2 v1 built them).

## 3. Model (`src/model.py` — no code changes)

`build_multiplane_model(backbone_name, n_findings=12, n_slots=18, ...)`
— the existing `n_slots` parameter (default 6) is simply passed 18.
`SlotAttentionModel.forward()` already asserts `slot_images.shape[1] ==
self.n_slots` and `slot_mask.shape == (B, n_slots)`, so it validates the
new shape automatically, no new guard code needed.

The per-finding attention query/head (`self.query`, `self.heads`) are
shaped by `embed_dim` only, not slot count — they need no changes to
support learning **independent attention weight per (slot, group) pair**
per finding, which is the entire point of this experiment: instead of us
pre-deciding which group matters, the model's existing masked-softmax
attention learns it from data.

One consequence worth stating explicitly: with 18 always-jointly-present-
or-absent pseudo-slots instead of 6, the "at least 1 present slot per
study" floor in `masked_finding_attention()` is no harder to satisfy
than before (a study with 0 present real slots still has 0 present
pseudo-slots, and vice versa — this is an equivalence, not a loosening) (the real corpus's measured minimum of 3 present slots per study
becomes a minimum of 9 present pseudo-slots) — no new failure mode here.

## 4. Training loop and compute

**Scope: fold 0 only**, same practice as A2 v1's own first run — compare
directly against A2 v1's fold-0 number (0.7689), not the pooled 0.7512,
since this is a same-fold, apples-to-apples comparison before deciding
whether to spend the other 3 folds.

**Hyperparameters: unchanged from A2 v1** (`AdamW`, `weight_decay=0.02`,
`lr_backbone=8e-6`, `lr_head=1e-3`, last 6 transformer blocks unfrozen,
12 epochs, OneCycle) — deliberately isolating `group_index`/pseudo-slot
count as the only changed variable, not re-tuning anything else at the
same time.

**Compute risk, not a design blocker:** the backbone now runs on 18
pseudo-slot images per study instead of 6 (~3x forward/backward compute
and activation memory; per-sample float32 input also grows from
`6*3*224²*4 ≈ 3.6MB` to `18*3*224²*4 ≈ 10.8MB`, worth an explicit host-RAM
check in the smoke test alongside the VRAM one, since several DataLoader
workers prefetching at that size is GB-scale). A2 v1's `batch_studies=32`
(already bumped from the sourced default of 8 based on real measured
VRAM headroom) may not fit at 3x memory.

**Fallback, if so: gradient accumulation to hold the *effective* batch
at 32** (e.g. `micro_batch=8` x `accumulate_steps=4`), not a smaller
real batch size. §4's premise is that `group_index`/pseudo-slot count is
the *only* changed variable versus A2 v1 — silently shrinking the batch
size would also change the OneCycle schedule's step count and the
effective LR dynamics, confounding the comparison. Handle the
micro-batch/accumulation split empirically in the notebook based on
what the pre-flight smoke test's VRAM headroom allows, and record
whichever split actually worked in the notebook's real-output section —
not pre-decided here. **If accumulation is used, the LR scheduler must
step once per accumulation cycle, not once per micro-batch** — `05v2`
sets `OneCycleLR(..., total_steps=EPOCHS * len(train_loader))`; with
accumulation this becomes `EPOCHS * (len(train_loader) // accumulate_steps)`.
Stepping per micro-batch instead would silently reintroduce the schedule
change this fallback exists to avoid. (DINOv2's normalization layers are
LayerNorm, not BatchNorm, so the micro-batch split itself doesn't perturb
normalization statistics — only the scheduler needs this care.) Estimated wall-clock: ~2-2.5h for fold 0 at batch
32 (up from A2 v1's ~47min), more with accumulation overhead if needed —
well inside the 30h/week Kaggle quota either way.

**Also accepted, not fixed (YAGNI for a pilot):** zero-filled absent
slots still run through the backbone (the mask only affects pooling, not
which slots get encoded), so the real compute overhead is somewhat above
the 3x arithmetic above for studies missing slots. Correctness-neutral;
a mask-aware gather that skips absent slots is real future work if this
architecture graduates, not part of this pilot.

## 5. Evaluation and gating (`src/evaluate.py` — reused, not changed)

**This section was rewritten after a first Opus review found the
original version unreliable at fold-0 sample sizes** — both the
statistical design (a 0.03 per-finding tolerance is 5-10x below the real
Hanley-McNeil standard error at ~17 gold studies/fold, and one of the 4
gate findings, `oa_lateral_compartment`, has no fold-0 baseline at all —
it was undefined, single-class-column, in A2 v1's own fold 0) and a
literal bug (`per_label_gate`'s `broad_effect` flag is true for a
*broad move in either direction*, not specifically an improvement — it
does not check the sign of the change against "better"). Two tiers,
matching how A2 v1 itself only trusted its weak-finding numbers after
pooling all 4 folds, not from fold 0 alone.

### 5.1 Fold-0 pilot — a macro regression check plus a directional read, not a statistical verdict

**Baseline:** A2 v1's fold-0 gold per-finding AUC
(`notebooks/05v2_slot_attention_baseline.ipynb`'s real output). Note
precisely: **0.7689 is an 11-finding macro** — `oa_lateral_compartment`
was undefined (single class among fold 0's 17 gold studies) and excluded
from the mean by `macro_roc_auc()`, not a typo carried from A2 v1's own
memory record.

**Macro check (the only pass/fail-style check at this tier):** compute
`candidate_gold_macro` the same way (11-finding mean, same exclusion
rule). Regression check: `candidate_gold_macro - 0.7689 >= -0.03`
(`gold_tol=0.03`, A0's own established range for macro-level seed
variance — this tolerance is legitimate at the *macro* level, since
averaging over 11 findings reduces noise; it is only unreliable applied
to a single finding at fold-0 sample sizes, which is why it is not used
per-finding below). **Caveat:** `0.02-0.05`'s source (`worse_of_two`'s
own docstring) measured *pooled*-OOF-macro seed variance, which is lower
noise than a single 17-gold-study fold's macro — `0.03` here is a
reasonable but not perfectly matched borrow (A2 v1's own real
epoch-to-epoch spread near its fold-0 peak was ~0.012, some supporting
evidence it's not wildly off). Since this is a one-sided **stop** gate
(a too-loose tolerance risks a false *graduation*, a too-tight one risks
a false *stop*), a marginal breach in the `-0.03` to `-0.05` range should
be treated as "borderline, use judgement" rather than an automatic hard
stop. **Not** `worse_of_two()` — that function requires a
weak-label gauge (`baseline_weak`, `candidate_weak`, `weak_tol`) with no
sound default; A2 v1's own weak gauge was itself flagged
"not directly comparable" (scored on only 32/1,290 weak studies). Rather
than invent a `weak_tol`, this pilot checks the gold macro only, stated
explicitly as a narrower check than `worse_of_two`'s usual two-gauge
design.

**Directional read (informative, not a hard gate):** report the
per-finding delta for **`medial_meniscus_tear`** and
**`lateral_meniscus_tear`** only — the two weak-cluster findings with a
real, defined fold-0 baseline (0.458 and 0.652 respectively) *and* a
per-finding weak-label AUC that was already good (0.9483/0.8789,
2026-08-28 investigation), so their fold-0 AUC is a genuine read of the
architecture, not diluted by label noise. **`mcl_injury` and
`oa_lateral_compartment` are excluded from this fold-0 read** —
`oa_lateral_compartment` for lacking a baseline (above); `mcl_injury`
for two compounding reasons: (a) its fold-0 baseline (0.933, on ~2
positives) has the widest noise band of any finding in this comparison,
and (b) `08v1`'s real Kaggle run found **2 of 9 (22%) gold
`mcl_injury`-positive studies have a completely blank `COR_T1` slot** —
for those studies, adding groups 0/2 of an absent slot adds nothing, an
intrinsic cap on how much this specific change can move this specific
finding. Do not compute or quote a `per_label_gate()` `broad_effect`
verdict at this tier — with only 2 usable findings and single-fold noise
this size, a fixed-tolerance broad-effect check would be reading noise,
same failure mode the first review caught. Just report the two real
deltas and their direction.

**Decision rule for whether to scale to 4 folds:**
- Macro check fails (regression `> 0.03`) → **stop, do not scale.**
  Report the real result; treat hypothesis 2 as not supported by this
  pilot (§8's `RSNA_WINDOW`-widening alternative remains open, at
  resolution-fix cost).
- Macro check passes **and** both `medial_meniscus_tear` and
  `lateral_meniscus_tear` move in the positive direction by a margin
  that looks real rather than noise-sized (context: this fold's
  Hanley-McNeil SE for these two findings is 0.144/0.141 respectively —
  treat a move under ~0.10, i.e. under one point-estimate SE, as
  inconclusive, not evidence either way, per §1's stated-up-front
  small-expected-effect prior. Using a bar below the raw single-point SE
  is deliberate, not an oversight: a *delta* between two correlated
  models trained on the same fold/data has a smaller SE than either
  model's own point estimate, and the only cost of a false positive here
  is spending 3 more folds where §5.2's real statistical gate lives — not
  a wrong graduation decision) → **scale to the
  remaining 3 folds.**
- Macro check passes but the two findings don't move positively, or move
  by an amount indistinguishable from single-fold noise → **stop,
  report as inconclusive** (not "disproved" — fold 0 alone cannot
  distinguish a real small effect from noise at this finding's sample
  size; a genuinely small effect, consistent with §1's prior, would look
  identical to this outcome). Scaling to 4 folds anyway is a judgement
  call for the user at that point, not automatic.

### 5.2 Pooled 4-fold gate — only if 5.1 says to scale

The statistically meaningful check, matching A2 v1's own precedent
(fold-0 anomalies for `medial_meniscus_tear`/`oa_lateral_compartment`
were only resolved by pooling). Once all 4 folds are trained and pooled
gold OOF predictions exist for all 58 gold studies:

1. **`per_label_gate(baseline_auc, candidate_auc, tol=0.03,
   min_concordant=3)`** on the **original 4-finding weak cluster**
   (`mcl_injury`, `oa_lateral_compartment`, `medial_meniscus_tear`,
   `lateral_meniscus_tear` — all 4 have real, defined pooled baselines
   at this tier: 0.6145 / 0.6325 / 0.6635 / 0.6584 respectively, unlike
   fold 0). **Decision rule, explicit (fixing the sign bug found in
   review): treat the hypothesis as supported only if `broad_effect` is
   true `AND macro_delta > 0`** — `per_label_gate` alone reports a broad
   *move*, not a broad *improvement*; a uniform regression across the
   cluster would otherwise pass `broad_effect` incorrectly.
2. **Gold macro regression check**, same construction as §5.1 but on the
   pooled 12-finding macro against 0.7512 (A2 v1's pooled baseline) —
   again a direct comparison with `gold_tol=0.03`, not `worse_of_two()`,
   for the same weak-gauge-comparability reason as §5.1.

**Graduation decision:** both checks pass → graduate `expand_slot_groups()`
to `src/features.py` and `SlotCacheDataset`'s `expand_groups=True` path
to `src/dataset.py` (per the notebook-before-src rule, §6). Either check
fails → do not graduate; report the real result and treat hypothesis 2
as not empirically supported at this scale.

## 6. Testing strategy

**Unit tests** (hermetic, synthetic fixtures, no Kaggle/GPU needed),
following A2 v1's existing test patterns in `tests/`:
- `expand_slot_groups()`: on a small synthetic `(6, 9, H, W)` array,
  verify output shape `(18, 3, H, W)`, verify every pseudo-slot's pixel
  content against `select_group(stack[s], g)` per §2.1's ordering
  equation exactly (no silent transpose/ordering bug), verify the
  mask-replication logic (`(6,)` -> `(18,)`, each real slot's bit
  repeated exactly 3 times at indices `s*3, s*3+1, s*3+2`).
- `SlotCacheDataset(..., expand_groups=True)`: real-shaped synthetic
  shard, confirm it returns `(18, 3, H, W)` images / `(18,)` mask;
  separately confirm `expand_groups=False` (the default) still returns
  A2 v1's exact `(6, 3, H, W)` shape and existing
  `tests/test_dataset.py` cases still pass unmodified; confirm
  `expand_groups=True, group_index=0` raises, per §2.2's exact rule.
- `build_multiplane_model(n_slots=18, pretrained=False)` constructs
  without error and a forward pass on synthetic `(B, 18, 3, H, W)` +
  `(B, 18)` inputs produces `(B, 12)` finite logits — cheap, hermetic,
  confirms A2 v1's existing shape-assertion guards correctly accept the
  new slot count (no new guard code to test, since none was added).

**Kaggle validation notebook** (self-contained, no `import src`, same
convention as `05v2`/`06v2`/`07v1`/`08v1`/`08v2` — exact filename decided
at implementation-plan time, e.g. `09v1_a2v2_multigroup_baseline.ipynb`):
1. **Load `fold_assignments.csv` from `05v2`/`06v2`'s existing Kaggle
   output** (saved specifically "so a future notebook doesn't need to
   redo this scan" — its own words) rather than regenerating fold
   assignments from a fresh 4,407-study DICOM header scan. Assert fold
   0's val set matches what was already recorded (1,307 val studies, 17
   gold) before trusting any comparison against 0.7689 — a silent fold
   mismatch would invalidate §5.1 without failing anything.
2. Build `expand_slot_groups()`'s notebook-local copy; sanity-check a
   few real batches (shapes, no NaN, mask-replication matches the source
   6-slot mask, per §6's unit-test equations).
3. Overfit-a-tiny-batch smoke test (same practice as A2 v1), including
   the VRAM **and host-RAM** checks and the gradient-accumulation
   fallback described in §4.
4. Full fold-0 training run, §4's hyperparameters (micro-batch/
   accumulation split as resolved by the smoke test).
5. Report real per-finding gold AUC, the §5.1 macro regression check and
   directional read on `medial_meniscus_tear`/`lateral_meniscus_tear`,
   and the resulting scale-or-stop decision — computed for real, not
   estimated.
6. **Only if step 5 says to scale:** train folds 1-3, pool, and run
   §5.2's real `per_label_gate`/macro-regression checks for the actual
   graduation decision.

Only after the user reviews this notebook's real output does any of this
graduate to `src/` (`expand_slot_groups()` → `src/features.py`,
`SlotCacheDataset`'s `expand_groups=True` path → `src/dataset.py`) — same
notebook-before-src discipline as every other item in this project.

## 7. Process note

Per the user's explicit request (2026-08-28,
[[feedback-opus-review-spec-before-implementing]]): this spec gets an
Opus-model review **before** the implementation plan is written/executed
— not only a post-implementation review. The review should check this
spec against the real evidence in `08v1`/`08v2`'s Kaggle output and
`RESOURCES.md`'s cited sources, not just internal consistency.

**First review pass (2026-08-28):** found §5's gate design unreliable at
fold-0 sample sizes and one real bug (`per_label_gate`'s `broad_effect`
not sign-checked) — all verified against the real code/data, not just
argued. §5, §2.2, §4, and §6 were rewritten in response.

**Second review pass (2026-08-28), on the revision:** verdict "sound" —
re-derived all 8 fixes independently against the real code/data (not
just re-reading the spec's own claims) and confirmed each landed
correctly. Found one real wording error (§2.2 had said `group_index` was
"required," when it is optional with default `1` — an implementer
following that sentence literally would have broken the exact tests the
fix exists to protect) plus four minor clarifications (the
gradient-accumulation scheduler-stepping detail in §4; the
`expand_groups=True, group_index=0` exact test condition in §2.2/§6; the
reasoning behind §5.1's sub-SE "looks real" bar; a caveat that §5.1's
`gold_tol=0.03` is borrowed from a pooled-level measurement, noisier
applied to a single fold). All fixed inline in this revision — **spec
approved, ready for the implementation plan.**

## 8. Open items carried forward, not blocking this spec

- The resolution fix (224px → 336px A3 cache rebuild) — deferred,
  separate decision, independent of this spec's outcome either way.
- Scaling from fold 0 to the full 4-fold CV — only after fold 0's gate
  passes (§5).
- `synovitis`'s RadImageNet-domain-pretraining lead (from the live forum
  search, `RESOURCES.md`) — unrelated to this spec, a separate future
  item.
- If this spec's hypothesis does *not* hold up empirically: the
  `RSNA_WINDOW=0.35,0.65` narrow-sampling-window explanation (still
  standing, unweakened by the `08v2` control) would need a cache rebuild
  to actually test (widening the window, not just which of the existing
  3 groups gets used) — same cost class as the resolution fix, a later
  decision.
