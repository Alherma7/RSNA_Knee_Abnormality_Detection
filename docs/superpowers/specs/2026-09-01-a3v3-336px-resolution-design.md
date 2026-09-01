# A3 v3 — 336px cache rebuild with the wide window folded in, design

**Status:** revised after a first Opus review found 3 Critical + 8
Important + 8 Minor findings (all addressed below). Ready for the
implementation plan.
**Plan item:** the "224px→336px resolution rebuild" lead named as
explicitly out of scope in A3 v2's own design (§5, "the resolution fix
stays a separate, later, conditional decision") — chosen now, with the
previously-parked wide window bundled in, over RadImageNet pretraining
and ROI/spatial-attention pooling (both still on the table if this
doesn't move the needle).
**Depends on:** the vendored reference kernel
(`data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`,
`RESOURCES.md`'s `stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache`
entry), the A3 v2 cache-build notebook
(`notebooks/13v1_a3v2_window_cache_build.ipynb`) as the direct template
for §2, the A2 v2/A2 v3 pilot notebooks
(`notebooks/09v1_a2v2_multigroup_baseline.ipynb`,
`notebooks/14v1_a2v3_window_only_fold0_pilot.ipynb`) as the direct
template for §3, A2 v2's real fold-0 result (0.7956) as the gate
baseline.

## 1. Overview

A2 v2's pooled 4-fold result (0.8009, real LB 0.849) still leaves a
5-finding cluster (`mcl_injury`, `oa_lateral_compartment`,
`medial_meniscus_tear`, `lateral_meniscus_tear`, `synovitis`) stuck at
0.61-0.66 pooled gold AUC against a healthy cluster at 0.79-0.88.
`synovitis` has a separately-diagnosed, structural label-noise cause
(RESOURCES.md, `starkhushi`'s notebook) not addressed here. The other
four are consistent with a shared "small/thin/focal finding lost to
input resolution or global pooling" mechanism, independently observed
by two external competitors on two different architectures:
`wguesdon`'s Nyquist-sampling analysis for DINOv2 ViT-S/14 (a
patch-token mechanism; 448px was also tested there and showed no
further gain) and `dreaddevelopment`'s CoAtNet pipeline (a CNN, no
patch tokens, no 448px test in that source). The two converge on the
same **336px** number via two different mechanisms, not two citations
of the same argument — if anything stronger evidence than either alone
(RESOURCES.md, both citations).

**The wide window (A3 v2).** A3 v2's own fold-0 pilot (`14v1`) tested
widening `RSNA_WINDOW` from the narrow 35-65% pin to the reference
kernel's own wider `PLANE_WINDOW` per-plane defaults, in isolation at
224px. Real result: macro passed (0.7726 vs. 0.7956, delta -0.0230,
inside tolerance) but neither meniscus finding cleared the pre-agreed
≥0.10 bar (`medial_meniscus_tear` -0.1942, `lateral_meniscus_tear`
+0.0601) — the pre-agreed I4 gate said **STOP**, not scaled, not
adopted standalone.

**2026-09-01 re-analysis (this project, no new Kaggle run):** that STOP
result was re-examined for whether the 58-gold gate's known noise could
have produced it. Two views of the same underlying sampling variance —
one observed, one analytic — corroborating each other, not two
independent sources:

- Fold 0's own 17-study readout of `medial_meniscus_tear` (0.458, A2 v1,
  `05v2`) sat **0.2055** below the trustworthy pooled 58-study readout
  (0.6635, `06v2`) — a direct measurement of how far a 17-study
  per-finding AUC can land from the fuller picture. (Not a controlled
  same-model comparison: the pooled figure combines OOF predictions
  from 4 different fold checkpoints across all 58 gold studies, not a
  re-score of fold 0's 17 under one fixed model — but it is still real
  evidence of how far a single 17-study fold's per-finding estimate can
  drift.) Almost identical in magnitude to the -0.1942 delta that
  triggered `14v1`'s STOP.
- A Hanley-McNeil back-of-envelope at the real fold-0 class split
  (n=17, 8 pos/9 neg — recoverable exactly from `09v1`'s own recorded
  AUC denominators, not approximate): SE(single fold-0 AUC) ≈ 0.144.
  Treating the A2 v2 baseline and the A3 v2 candidate as independent
  gives SE(delta) ≈ 0.204 — an **upper bound**, not a best estimate,
  since the two runs share the same 17 gold studies and most of the
  same pipeline, so their errors are plausibly positively correlated,
  which would shrink the true paired SE below 0.204. At the
  independence bound the observed -0.1942 delta gives z≈0.95 (p≈0.34,
  two-sided) — right at the edge of 1 SE, not comfortably inside it.
  Under plausible correlation (e.g. ρ=0.5) it rises to z≈1.35 (p≈0.18)
  — still short of conventional significance either way, but the data
  are equally consistent with a small *real negative* effect from the
  wide window as with pure noise. This asymmetry matters for the
  decision below and for I3 in §5.

**Decision (user, 2026-09-01):** don't re-run the window-only pilot to
grind down that noise in isolation. Since the wide window costs nothing
extra on top of any cache rebuild (A3 v2 §1's own finding — same slice
count decoded, same cache shape/size), **fold it into the 336px rebuild
as the default** and let that experiment's own gate (same design as
`14v1`'s, kept unchanged — see §4) decide the combined bundle. This
deliberately gives up attribution: a single treatment arm (336px + wide
window) measures only the *combined* effect against the (224px,
narrow) baseline — not either change's isolated contribution, and not
their interaction (estimating an interaction would need a third,
(336px, narrow) cell, which this spec does not build). Accepted because
the window's isolated test lacked the statistical power to resolve an
effect of the size actually expected, so re-isolating it again is
unlikely to be informative on its own — but this is not costless: the
window's own fold-0 point estimate was **negative**
(-0.1942 on the primary gating finding), so the joint arm is not "two
neutral changes," it is a resolution change stacked on a change whose
own measured direction (if real, not noise) points the wrong way.
Accepted anyway, in exchange for one cache rebuild instead of two — see
§5's disentangling note for what happens if this trade-off needs
revisiting after a gate result.

**This spec's scope:** rebuild the A3 slot cache at `img=336`,
`crop_mm=130` (unchanged — the reference kernel's own comment confirms
130mm@336px = 5.42mm/token, the cited "2.25x compute" figure, vs.
130mm@224px = 8.13mm/token, today's champion). Per the reference
kernel's own framing (`CFG` comment), `img` and `crop_mm` are two knobs
on one ratio (mm/token = `14*crop_mm/img`) — the same 5.42 target could
in principle be reached at 1.0x compute by shrinking `crop_mm` to
~87mm instead. Not adopted here: a ~87mm crop loses real field of view
(130mm was itself chosen to stop silently skipping the crop on ~61% of
studies whose FOV sat at or below the old 160mm threshold, per A3's own
history), so the resolution route is preferred despite its cost. **With
`RSNA_WINDOW` unset** (same wide `PLANE_WINDOW` default `13v1` already
validated — not re-deriving it, just carrying it forward). Same
`n_group_max=3`, same 6 named slots, same 9-slices/slot/3-anchor-group
structure as every prior A3 cache. The only two changed variables from
the *current, narrow-window, 224px* cache are `img` and `window`,
changed together — a deliberate departure from A3 v2's own
confound-avoidance stance (§1 there explicitly isolated window alone),
for the reasons stated in the decision above.

## 2. Cache rebuild

New notebook, CPU-only, `notebooks/15v1_a3v3_336px_cache_build.ipynb`,
directly modeled on `13v1`'s structure (same script, same two pre-build
gates). Runs the vendored `build_pixel_cache.py` with:

- `RSNA_CROP_MM=130` (unchanged)
- `RSNA_N_GROUP_MAX=3` (unchanged)
- `RSNA_IMG=336` (the resolution change — was unset/224 in every prior
  cache including `13v1`)
- `RSNA_WINDOW` **not set** (the window change — same as `13v1`, carried
  forward unchanged)

### 2.1 Fingerprint diff and probe gates

`13v1`'s Step 0 diffed the new build's `cache_meta.json` fingerprint
against the *current, narrow-window, 224px* cache and expected exactly
one field (`window`) to differ. This build changes **two** fields at
once, so the diff step must be adjusted: diff against the **original
narrow-window 224px cache** (`stevenleehans`'s published Dataset, still
the production cache A2 v2 depends on) and expect `window`
(`"0.35,0.65"` → `"default"`) **and** `img` (`224` → `336`) to differ,
with every other field (`crop_mm`/`pad_short_fov`/`lat_from_geometry`/
`group`/`n_group`/`slots`/`seed`) matching exactly, same as `13v1`'s own
check.

**Corrected (I5): the originally-proposed "diff `window` against
`13v1`'s cache" check is dropped — it was vacuous.** The `window` field
in `cache_meta.json` is set directly from
`os.environ.get("RSNA_WINDOW", "default")` (reference kernel, cell 25)
— it records only whether the override was set, not what
`PLANE_WINDOW`'s actual per-plane values were. Both `13v1` and this
build omit the override, so both would read `"default"` regardless of
whether `PLANE_WINDOW`'s contents had drifted between sessions — the
check cannot detect what it claimed to detect, and the real
override-was-set failure mode is already caught by the first diff
above. If drift-detection is wanted, the real check is printing
`PLANE_WINDOW`'s actual dict (or a hash of the copied `read_slot()`
cell) and asserting it against `13v1`'s recorded values — not attaching
`13v1`'s cache as a third Dataset input for a field comparison that
can't distinguish the two cases.

**Probe-rate acceptance criterion, corrected (I6) — an explicit band,
not a bare reference number.** `13v1` itself established the usable
band from the script's own commentary: **5-10/s is unremarkable, a
large deviation outside it (particularly below 5/s) is worth
understanding before the full run** — `13v1`'s own real result (7.0/s)
sat inside this band, not at the 9.0/s figure itself (that number was
measured at 224px and is separately documented as ~44% optimistic vs.
a real sustained rate). At 336px the same band is reused, adjusted for
an expected modest slowdown: run `--probe 40`, **accept 5-10/s** as
healthy, **stop and investigate below 5/s**. Decode cost (DICOM read +
`pixel_array`) is unchanged by output resolution — only the resize step
grows — so the drop from `13v1`'s 7.0/s, if any, is expected to be
modest, not proportional to the 2.25x pixel-count increase.

### 2.2 Cost, output-size limit, and multi-session shard split

**Corrected (C1) — the naive full build does not fit in one Kaggle
session's output, and the original draft did not address this.**
Storage scales with pixel count: `4407 studies × 6 slots × 9 slices ×
336² bytes ≈ 25.0 GiB` train-side alone (the 224px equivalent,
`4407×6×9×224² ≈ 11.12 GiB`, matches the real recorded 11.13 GiB
exactly, confirming the scaling). **25 GiB exceeds both the vendored
script's own guard** (`build_pixel_cache.py`: `if total_gb > 15: log("!!
... may exceed the Kaggle output limit")`) **and Kaggle's real
`/kaggle/working` output cap.** Running §2 as originally drafted (one
session, `--shards 4`) would burn the full ~140 min CPU estimate and
then fail to persist a usable artifact.

**Fix — more, smaller shards, built across (at least) two Kaggle
sessions, using the script's own existing resume/merge support** (not
new functionality — `main()` already accepts `shards=N`, a `shard`
subset argument like `"0,1"`, skips shards already present in
`out_dir`, and merges `cache_meta.json` across invocations with a
config-clash guard — the same mechanism `13v1` already exercises):

- `--shards 8` (up from 4) → ~3.1 GiB and ~551 studies per train shard,
  vs. ~2.8 GiB/1,101 studies at the current 4-shard/224px layout — both
  the per-shard output size and the in-RAM buffer `build_cache()` holds
  before `save_split()` roughly halve relative to a naive `--shards 4`
  run at 336px (~6.3 GiB would have been held in RAM per shard at
  `--shards 4`; `--shards 8` brings that back down to ~3.1 GiB, close to
  today's headroom).
- **Session 1:** build shards 0-3 (`--shard 0,1,2,3`), ~12.5 GiB output,
  under both guards. Download the output, re-upload as a new Kaggle
  Dataset.
- **Session 2:** attach session 1's partial output as an input Dataset
  at the same `out_dir` path the script expects, build shards 4-7
  (`--shard 4,5,6,7`) — the script's own skip-existing-shard logic
  leaves 0-3 untouched, and its `cache_meta.json` merge appends the new
  shards' entries to the ones already present (subject to the existing
  config-clash guard, which should report **no** clash since both
  sessions run identical `CFG` except by design).
- Test shards: unaffected in practice (the test split's placeholder
  studies are few; even at 336px the test-side output stays a small
  fraction of the train-side total) — build alongside session 2, or
  session 1 if simpler; not a sizing concern either way.
- Final step: download session 2's complete output (now covering all 8
  train shards + test shards + the merged `cache_meta.json`) and upload
  it as the definitive new cache Dataset. The current production cache
  and the parked A3 v2 cache both stay untouched and usable throughout.

**Cost estimate, restated:** ~2.25x `13v1`'s real build (61.3 min,
11.13 GiB — recorded in `project_rsna_phase_status.md`'s 2026-08-31
`13v1` update; `13v1`'s own "Real output" cell was left as a
placeholder and should be backfilled as a side task, same for
`README.md`) → **roughly ~70 min CPU per session** (half the corpus
each), **~25 GiB total split as ~12.5 GiB per session**, across (at
least) 2 sessions. Confirmed acceptable to the user in advance
(2026-09-01) specifically because this step is CPU-only and does not
touch the GPU-quota constraint that actually limits this project's
pace — that reassurance is scoped to §2 only, not to §3 (see I4 below).

**Downstream consequence, must be handled in §3:** `16v1`'s cache-
loading code must enumerate shard names from the new `cache_meta.json`
(as `src/data.py::load_slot_cache_shard()`'s own docstring already
recommends — "see `cache_meta.json` in `cache_dir` for the full list")
rather than hardcoding the `s00of04`..`s03of04` literal names every
prior notebook's hand-copied loading cell has used, since this cache
has 8 train shards, not 4.

## 3. Training pilot (fold 0 only)

New notebook, `notebooks/16v1_a2v4_336px_fold0_pilot.ipynb`, mirroring
`14v1`'s structure closely: fold-assignment re-assertion, dual
best-epoch/SWA checkpoint tracking with the gold-AUC readout for the
SWA branch, the SWA `.clone()` corruption hazard handled the same
deliberate way, the cache-identity guard concept — all of that
**carries forward from `14v1` in structure**, but four concrete sites
in `14v1`'s own template hardcode `224` and must be updated for this
notebook to run against a 336px cache at all, not just at graduation
time. **Corrected (C2): the original draft understated this as one
gap in `src/model.py`; it is actually four sites in the notebook
template itself, none of which route through `src/`.**

**3.1 The four `224` sites, checked directly against `14v1`'s real
cells:**

| site | `14v1`'s code | change needed in `16v1` |
|---|---|---|
| Cache-identity guard (`14v1` cell 10) | `_EXPECTED = {"window": "default", "crop_mm": 130.0, "img": 224, ...}` | `"img": 336` — otherwise this guard **hard-aborts** on the new cache before training starts |
| Backbone construction (`14v1` cell 13) | `timm.create_model(backbone_name, pretrained=True, num_classes=0, img_size=224)` | `img_size=336` |
| Model's own forward-shape check (`14v1` cell 13) | `if (S, C, H, W) != (self.n_slots, 3, 224, 224): raise ValueError(...)` | `(self.n_slots, 3, 336, 336)` |
| Dataset pre-flight assert (`14v1` cell 9) | `assert images.shape == (N_SLOTS, 3, 224, 224)` | `(N_SLOTS, 3, 336, 336)` |

**Correction to the original draft's causal claim:** the notebooks do
**not** call `src/model.py::build_backbone()` — none of them `import
src` at all (confirmed across `05v2`/`06v2`/`07v1`/`09v1`/`10v1`/`11v1`/
`12v1`/`14v1`); each hand-duplicates a `timm.create_model(...)` call
with `img_size` hardcoded inline. So the reason every prior notebook
has trained at 224px is that literal, not `SlotAttentionModel.__init__`
failing to forward a parameter — that gap is real, but it lives in
`src/model.py:112` and only matters if/when this graduates (§5), it is
not why any notebook has been at 224px to date. `16v1`'s own
hand-duplicated model class needs its `img_size=336` threaded through
the same way `14v1`'s did at 224 — a literal edit, not a missing-
parameter fix, since the notebook never called the `src/` function to
begin with. Also note: the notebook's hand-kept `forward()`
shape-check (table row 3) is *stricter* than `src/model.py`'s own
version (`src/model.py:129-133` checks only `S`/`C`, leaves `H`/`W`
unconstrained) — the "shape-agnostic, no changes needed" reassurance
below is about `src/`, not about the template being copied.

`select_group()`/`expand_slot_groups()`/`SlotCacheDataset` (`src/`)
need no changes — re-confirmed directly against current code: all three
operate on tensor shape read from the cache array itself, no literal
`224` in any control-flow path (the `224` appearing in
`src/dataset.py`/`src/data.py` is comment/docstring-only, describing
the current cache's shape, not a hardcoded dimension).

**3.2 Patch-grid note, restated more precisely (M1).** 336 is evenly
divisible by DINOv2's patch size 14 (24×24 patch grid), a requirement
for `timm`'s `PatchEmbed` to construct without error — but divisibility
is a validity precondition, not by itself a safety property of the
position-embedding interpolation (`timm` bicubically resamples the
learned 2D grid to whatever target grid it's given, divisible or not).
The more relevant point: DINOv2's *native* pretrained grid is 37×37
(518/14) — 336px's 24×24 target is a **smaller** reduction from that
native grid than the 16×16 (224px) this project already runs in
production, so the resample this spec introduces is, if anything, less
lossy than the one already validated, not more. Still the first time
this project runs DINOv2 at any resolution other than 224 — the
pre-flight smoke test below confirms this empirically rather than by
this argument alone.

**3.3 VRAM must be re-measured, with a concrete expected landing point
(M8, sharpened from a vague "re-measure").** `MICRO_BATCH=8`/
`ACCUMULATE_STEPS=4` was A2 v2's own real measured-safe split at
224px/18-pseudo-slots, using **6.73 GB of a 15.6 GB T4's 85% (≈13.3 GB)
threshold** (`09v1`'s own recorded pre-flight output). At 336px, tokens
per image rise 256→576 (a 2.25x input, and ViT attention cost is
superlinear in token count, so compute/activation memory rises by more
than 2.25x, not exactly 2.25x). Scaling the activation share by ~2.25x
alone already projects close to ~14.7 GB — over the 13.3 GB threshold.
**Expected landing point: `MICRO_BATCH=4`/`ACCUMULATE_STEPS=8`**
(halved micro-batch, doubled accumulation, same effective batch size
32 preserved) — stated here as the concrete hypothesis the pre-flight
smoke test (overfit 8 real studies, shape/NaN/VRAM guards, same
practice as every prior training notebook) must confirm or correct
empirically before any full-epoch training runs. Report whatever the
real safe split turns out to be in the notebook's own "Real output"
section.

**Otherwise unchanged from `14v1`:** `AdamW`, `lr_backbone=8e-6`,
`lr_head=1e-3`, `weight_decay=0.02`, last 6 transformer blocks unfrozen,
12 epochs, OneCycle, `swa_epochs=3`, a pinned seed (matching `14v1`'s own
correction), `expand_groups=True` (18 pseudo-slots, A2 v2's winning
architecture).

**GPU cost, priced explicitly (I4) — this step, unlike §2, does draw on
the project's binding constraint.** A2 v2's fold-0 pilot ran ~2-2.5h,
"well inside the 30h/week Kaggle quota" (project memory). At ≥2.25x
tokens per image, plus whatever `ACCUMULATE_STEPS` doubling costs in
wall-clock, this pilot is expected to land around **~5h**. If §4's gate
says scale, the 4-fold run would be roughly **~18-22h of the 30h weekly
quota** — a materially larger commitment than A2 v2's own 4-fold run,
worth having explicit before committing, unlike §2's cache build which
genuinely doesn't touch this constraint.

## 4. Gate (fold-0 pilot only — unchanged from `14v1`'s design, confirmed with the user 2026-09-01)

Kept identical to `14v1`'s gate on methodological-consistency grounds
(the user's explicit 2026-09-01 choice, weighed against a
cluster-directional-consistency alternative and a skip-straight-to-4-fold
alternative — both considered, neither adopted, see chat record) even
though §1's re-analysis shows the individual ≥0.10 bar sits close to
this fold's own noise floor. Accepted: this pilot may again land
"inconclusive" rather than resolve the question outright — the point of
keeping the same gate is comparability with `14v1`'s own result and
avoiding a bespoke gate invented post-hoc for this one experiment.

**Baseline:** A2 v2's real fold-0 result, **0.7956, an 11-finding macro**
(`oa_lateral_compartment` undefined, single-class, among fold 0's 17
gold studies — unchanged from every prior fold-0 pilot). **Restated
limitation, carried forward from A3 v2's own spec (M4): 0.7956 is a
single unseeded draw** (`09v1` calls neither `torch.manual_seed` nor
`np.random.seed`), while `16v1` pins a seed per §3's own convention —
comparing a seeded candidate against an unseeded baseline remains an
accepted, not resolved, limitation, same as it was for `14v1`.

**Macro check (pass/fail):** `candidate_gold_macro - 0.7956 >= -0.03`
(`gold_tol=0.03`, same tolerance, same borderline-is-judgement-call
framing as `09v1`/`14v1`).

**Directional read:** `medial_meniscus_tear` (baseline 0.597) and
`lateral_meniscus_tear` (baseline 0.652) remain the two gating findings.
`mcl_injury` (baseline 0.800) reported as context only, not gating —
unchanged reasoning from `14v1`'s own corrected framing. `synovitis` not
gated here — it's the separately-diagnosed label-noise finding (§1), not
expected to move on a resolution/window change and not part of this
hypothesis.

**Non-gating secondary readout, new (I8):** alongside the two gating
findings, also report the **mean delta across the 4 weak-cluster
findings** (`mcl_injury`, `medial_meniscus_tear`, `lateral_meniscus_tear`,
`oa_lateral_compartment` where defined). Under the same independence-
bound assumption as §1, this mean has SE ≈ 0.204/2 ≈ 0.10 — roughly 2x
more sensitive than any single finding, and the mechanism under test
(resolution helping small/focal findings) predicts a *shared*
directional move across exactly this cluster, not necessarily a large
move in any one finding. **Explicitly non-gating** — does not change
§4's decision rule below, purely reported context so a likely-
inconclusive run per-finding still yields some real signal.

**Decision rule (pre-agreed, not relitigated after seeing the result):**
- Macro check fails → **stop**, report negative, do not scale.
- Macro check passes **and at least one** of `medial_meniscus_tear` /
  `lateral_meniscus_tear` moves positively by more than ~0.10 (same
  Hanley-McNeil-derived bar as `14v1`, unchanged), `mcl_injury`'s delta
  reported alongside → **scale to the remaining 3 folds.**
- Macro check passes but **neither** finding clears the bar → stop,
  report as inconclusive (not disproved) — same framing as `14v1`, now
  with §1's own quantitative noise analysis already on record as the
  reason this framing is deliberate, not a hedge.

Both readouts (best-epoch and SWA) get evaluated and reported, same as
`14v1`; the gate itself compares the **best-epoch** number against
0.7956 for the same apples-to-apples reason `14v1` established.

## 5. Explicitly out of scope for this spec

- **Scaling to the full 4-fold pooled gate** — designed only if/when the
  fold-0 pilot's gate (§4) says to scale, matching the established
  two-tier precedent exactly.
- **Disentangling window from resolution, if it turns out to matter
  (I3).** Because §1 bundles both changes, a gate pass here cannot say
  whether resolution, window, or their interaction drove it, and a gate
  fail cannot say whether 336px alone would have passed had the window
  (with its own negative fold-0 point estimate) not been included. If
  §4's gate **passes**, the designated tie-breaker before paying the
  `11v1` submission-port cost below is one additional, cheaper
  disentangling cell: an isolated **(336px, narrow-window)** cache +
  fold-0 pilot, reusing `13v1`'s already-existing **(224px,
  wide-window)** result and `14v1`'s **(224px, narrow)** baseline to
  complete the 2x2 without a fourth build. Not designed here — a
  follow-up decision, not a default action, only relevant if the joint
  gate passes and the two changes' separate contributions become
  worth knowing before committing further.
- **Graduating anything to `src/`.** If this pilot (or its eventual
  4-fold scale-up) is adopted, three real changes are needed, not one
  (M5, expanded from the original draft): `src/model.py::
  SlotAttentionModel.__init__` needs to accept and forward an
  `img_size` parameter to `build_backbone()`; `src/model.py::
  build_multiplane_model()` (the public factory) needs the same
  parameter added, since it has none today; `src/config.py::
  SLOT_CACHE_IMG_SIZE` (currently hardcoded `224`) needs updating or
  parameterizing; and the `(6, 9, 224, 224)`-shape docstrings in
  `src/dataset.py`/`src/data.py` need updating to match, since they are
  the stated contract even though the code itself doesn't hardcode the
  dimension. Real, priced work, deferred to that actual graduation
  decision — not designed here, same discipline as every prior pilot in
  this project.
- **Porting the wide window and 336px resolution into the live-decode
  submission path** (`notebooks/11v1_a2v2_submission_inference.ipynb`) —
  needed only if this graduates to a real submission. `11v1`'s current
  `WINDOW = (0.35, 0.65)` uniform tuple would need converting to a
  per-plane lookup (A3 v2's own §5 already priced this), and its decode
  step would need the 336px crop/resize path added.
  **Feasibility flag, new (I7) — this deferral needs a sanity-check
  now, not just a cost estimate later.** `11v1`'s own real dry-run
  measured **4.612 s/study at 224px** (18 pseudo-slots × 4 folds)
  against a **≤9h** scored-rerun budget for the whole hidden test set,
  and its own code already flags this as worth watching. A competitor's
  independently-run **336px** pipeline is recorded (RESOURCES.md) at
  "94 inference windows/study costs ~9h for the full test set" — i.e.
  already at the wall others report for a comparable resolution. At
  ≥2.25x per-study cost plus a larger decode-side resize, a 336px port
  could exceed the 9h budget outright. Not designed here, but before
  spending a ~18-22h 4-fold scale-up (§3) on the strength of a passing
  fold-0 gate, do a back-of-envelope check (hidden-test study count ×
  projected 336px per-study cost vs. 32,400s) so an infeasible
  submission path is known *before* that spend, not discovered after
  it, with a stated fallback (fewer ensemble members, fp16/
  `torch.compile`, reduced slot count at inference) if the envelope
  doesn't fit.
- **RadImageNet pretraining and ROI/spatial-attention pooling** — the two
  other leads considered for the same weak-finding cluster, not chosen
  this round, still on the table if this pilot's result doesn't move the
  needle enough.
- **The `synovitis` label-noise problem** — a distinct, separately
  diagnosed issue (RESOURCES.md, `starkhushi`), not addressed by
  resolution or window changes.

## 6. Testing strategy

No `src/` changes in this spec (§3/§5 — the resolution/window changes
are notebook-only until a real graduation decision), so no new unit
tests are owed. The existing suite (90/90 passing, re-confirmed during
this spec's own Opus review) is not expected to change. Validation is
entirely real-Kaggle-output-based, same discipline as every prior item:

1. `15v1`: the two-field fingerprint diff (§2.1) run and reported before
   the probe; `--probe 40`'s measured rate checked against the explicit
   5-10/s band (not a bare comparison to 9.0/s); each session's build
   reports `complete: N/N shards` for the shards it covers; the final
   merged `cache_meta.json` reads `img: 336`, `window: "default"`,
   8 train shards recorded.
2. `16v1`: the cache-identity guard (with `img: 336`, §3.1) and
   fold-assignment re-assertion before training; shard names read from
   the new `cache_meta.json` rather than the old 4-shard literal
   convention (§2.2); the pre-flight smoke test confirms DINOv2 actually
   trains at 336px on real data (shape/NaN/VRAM guards, §3.3) and
   reports the real safe `MICRO_BATCH`/`ACCUMULATE_STEPS` split before
   any full-epoch training runs.
3. `16v1`'s own "Real output" section filled in with the real per-finding
   table (both best-epoch and SWA readouts), the §4 macro check, the
   directional read including `mcl_injury`'s context delta and the
   non-gating weak-cluster mean delta, and the resulting scale-or-stop
   decision per the pre-agreed rule — computed for real, not estimated.

## 7. Process note

Per [[feedback-opus-review-spec-before-implementing]]: this spec gets an
Opus-model review before the implementation plan is written, same
process as every prior architecture spec in this project.

**First review pass (2026-09-01):** confirmed every prior-run number
cited (0.7956/0.597/0.652/0.800 fold-0 baselines, `14v1`'s real delta
numbers, `13v1`'s real build stats, A2 v1's 0.458/0.6635 pair, the
Hanley-McNeil arithmetic to three decimals) matches the project's
committed record exactly, and every reference-kernel claim (`RSNA_IMG`/
`RSNA_CROP_MM` defaults, the `PLANE_WINDOW`/`RSNA_WINDOW` override
order, the cost-comment figures) verbatim-accurate against the actual
notebook JSON. Found 3 Critical issues — (C1) the ~25 GiB naive build
output exceeds both the vendored script's own 15 GiB guard and Kaggle's
output cap, with no split strategy in the original draft; (C2) §3's
claim that `14v1`'s template "carries forward unchanged" was false —
four hardcoded-`224` sites (a cache-identity guard, a backbone
construction call, a forward-shape check, a dataset pre-flight assert)
would abort or misconfigure the pilot, none of them routing through the
`src/model.py:112` gap the original draft blamed; (C3) the claim that
this design would resolve a window/resolution interaction was both
logically impossible (a single bundled treatment arm cannot estimate an
interaction) and misattributed (the cited "open question" was actually
A3 v2's own spec §4, about `mcl_injury`, not `14v1`) — plus 8 Important
findings (an overstated "independent lines of evidence" framing for the
noise re-analysis; an SE framing that called the independence bound
"conservative" when it is actually the permissive direction for the
spec's own conclusion; an understated asymmetry in bundling a resolution
change with a window change whose own point estimate was negative, with
no disentangling exit; the GPU-quota reassurance scoped too broadly to
cover §3 as well as §2; a "sanity check" in §2.1 that could not detect
what it claimed to; a probe-acceptance criterion with a comparand but no
actual pass/fail band; a submission-port feasibility risk deferred
without even a back-of-envelope check, against real evidence a
comparable 336px pipeline already sits at the scoring-time wall; and no
non-gating secondary readout for a pilot design already expected to
often land inconclusive) and 8 Minor findings (an overclaimed
patch-divisibility safety argument; an over-attributed 448px citation;
weakly-traceable build-cost figures; a dropped seed-limitation caveat;
an incomplete graduation-cost list; overstated independence of the two
noise-evidence lines; an unaddressed cheaper alternative route to the
same mm/token ratio; an under-specified VRAM projection). **All
addressed above** (§1 revised: corrected evidence framing, SE bound
reframed as an upper bound with a correlation-sensitivity note, the
interaction claim removed and corrected, the crop_mm alternative
addressed, the bundling trade-off stated honestly; §2 revised: the
vacuous sanity check dropped, the probe band made explicit, a real
multi-session/8-shard build plan added; §3 revised: all four `224` sites
enumerated with exact fixes, the causal claim about `src/model.py`
corrected, the patch-grid argument sharpened, a concrete VRAM
projection and expected micro-batch/accumulation split added, GPU cost
priced explicitly; §4: the unseeded-baseline caveat restated, a
non-gating weak-cluster mean-delta readout added; §5: a disentangling
tie-breaker step named, the graduation-cost list expanded to 3 items,
the submission-port feasibility flag added; §6 updated to match).
