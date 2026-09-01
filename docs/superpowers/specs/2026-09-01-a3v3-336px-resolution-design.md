# A3 v3 — 336px cache rebuild with the wide window folded in, design

**Status:** draft, awaiting Opus review (per
[[feedback-opus-review-spec-before-implementing]]).
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
by two external competitors on two different architectures
(`wguesdon`'s Nyquist-sampling analysis for DINOv2 ViT-S/14, and
`dreaddevelopment`'s CoAtNet pipeline) — both converge on **336px** as
the resolution where 1-3mm meniscal lesions survive patch-token
resizing, with no further gain measured at 448px (RESOURCES.md, both
citations).

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
have produced it. Two independent lines of evidence, both already
in-hand:
- The exact same fold-0 gold set (17 studies), same finding
  (`medial_meniscus_tear`), tracked across A2 v1's own fold-0-only
  (0.458) vs. pooled-4-fold (0.6635) readout — a **+0.2055** swing from
  pure fold-sampling noise alone, zero model change. Almost identical in
  magnitude to the -0.1942 delta that triggered `14v1`'s STOP.
- A Hanley-McNeil back-of-envelope at n≈17 (~8 pos/9 neg): SE(single
  fold-0 AUC) ≈ 0.144, SE(delta between two fold-0 measurements,
  conservative independent bound) ≈ 0.204 — the observed -0.1942 delta
  is well within 1 SE of zero (z≈0.95, p≈0.34), not statistically
  distinguishable from no effect.

**Decision (user, 2026-09-01):** don't re-run the window-only pilot to
grind down that noise in isolation. Since the wide window costs nothing
extra on top of any cache rebuild (A3 v2 §1's own finding — same slice
count decoded, same cache shape/size), **fold it into the 336px rebuild
as the default** and let that experiment's own gate (same design as
`14v1`'s, kept unchanged — see §4) decide both questions together. This
also directly answers `14v1`'s own explicitly-flagged open question:
whether widening the *coronal* window (0.35-0.65 → 0.15-0.85, which also
touches the `COR_FLUID_FS` slot) interacts with anything resolution
changes — it will now be tested inside the same run, not as a separate
confound to chase later.

**This spec's scope:** rebuild the A3 slot cache at `img=336`,
`crop_mm=130` (unchanged — the reference kernel's own comment confirms
130mm@336px = 5.42mm/token, the cited "2.25x compute" figure, vs.
130mm@224px = 8.13mm/token, today's champion), **with `RSNA_WINDOW`
unset** (same wide `PLANE_WINDOW` default `13v1` already validated —
not re-deriving it, just carrying it forward). Same `n_group_max=3`,
same 6 named slots, same 9-slices/slot/3-anchor-group structure as every
prior A3 cache. The only two changed variables from the *current,
narrow-window, 224px* cache are `img` and `window`, changed together —
a deliberate departure from A3 v2's own confound-avoidance stance
(§1 there explicitly isolated window alone), justified because §1's
noise re-analysis showed the isolated window test could not have
resolved a real effect of the size actually expected, so there is
nothing left to lose by testing them jointly, and real cost (a second
full CPU rebuild) to save by not re-isolating window alone again.

## 2. Cache rebuild

New notebook, CPU-only, `notebooks/15v1_a3v3_336px_cache_build.ipynb`,
directly modeled on `13v1`'s structure (same script, same two pre-build
gates, same shard/upload convention). Runs the vendored
`build_pixel_cache.py` with:

- `RSNA_CROP_MM=130` (unchanged)
- `RSNA_N_GROUP_MAX=3` (unchanged)
- `RSNA_IMG=336` (the resolution change — was unset/224 in every prior
  cache including `13v1`)
- `RSNA_WINDOW` **not set** (the window change — same as `13v1`, carried
  forward unchanged)

### 2.1 Fingerprint diff — two expected fields this time, not one

`13v1`'s Step 0 diffed the new build's `cache_meta.json` fingerprint
against the *current, narrow-window, 224px* cache and expected exactly
one field (`window`) to differ. This build changes **two** fields at
once (`window` and `img`), so the diff step must be adjusted:

- Diff against the **original narrow-window 224px cache**
  (`stevenleehans`'s published Dataset, still the production cache A2 v2
  depends on): expect `window` (`"0.35,0.65"` → `"default"`) **and**
  `img` (`224` → `336`) to differ, every other field
  (`crop_mm`/`pad_short_fov`/`lat_from_geometry`/`group`/`n_group`/
  `slots`/`seed`) must match exactly, same as `13v1`'s own check.
- **New, additional sanity check not present in `13v1`:** also diff the
  `window` field alone against `13v1`'s own already-built cache
  (`cache_meta.json`, from the parked A3 v2 build) — both should read
  `"default"` identically. This isolates whether the wide-window logic
  itself has drifted between the two build sessions, independent of the
  resolution change, at effectively zero extra cost (the field is
  already being read for the first check).

Probe-rate acceptance criterion: same corrected rate-based check as
`13v1` (`--probe 40`, compare measured slot-series/s against the
documented 9.0/s baseline, not a wall-clock projection) — the probe's
measured rate is expected to be **slower** than `13v1`'s own 7.0/s
result (more pixels decoded per slot at 336px), so the acceptance bar is
on the *documented* 9.0/s reference, not on matching `13v1`'s own
already-slower-than-reference number.

### 2.2 Cost and shard layout

Expected, not yet measured: ~2.25x `13v1`'s real build (61.3 min,
11.13 GiB) → **roughly ~140 min CPU-only, ~25 GiB**, since both compute
and storage scale with pixel count
(`(336/224)² = 2.25`). Confirmed acceptable to the user in advance
(2026-09-01) specifically because this is CPU-only and does not touch
the GPU-quota constraint that actually limits this project's pace.
Same shard convention as every prior cache (`--shards 4`, train + test),
uploaded as a new, separate Kaggle Dataset — the current production
cache and the parked A3 v2 cache both stay untouched and usable.

## 3. Training pilot (fold 0 only)

New notebook, `notebooks/16v1_a2v4_336px_fold0_pilot.ipynb`, mirroring
`14v1`'s structure closely (cache-identity guard, fold-assignment
re-assertion, dual best-epoch/SWA checkpoint tracking with the gold-AUC
readout added for the SWA branch, the `.clone()` corruption hazard
handled the same deliberate way) — **all of that carries forward from
`14v1` unchanged**, not re-derived here. Two things are genuinely new:

**3.1 `img_size` must be threaded through the hand-duplicated model
class — a real gap, checked directly against current code.**
`src/model.py::build_backbone()` already accepts an `img_size` parameter
and interpolates DINOv2's native 518×518 position embeddings down to it
(confirmed working at 224px, A2 v1's real Kaggle run). But
`SlotAttentionModel.__init__` calls `build_backbone(backbone_name,
pretrained=pretrained)` **without forwarding `img_size`** — so every
existing notebook (`05v2` through `14v1`) has always trained at the
default 224 regardless of cache resolution, because the cache itself
was always 224px until now. `16v1`'s own hand-duplicated copy of
`SlotAttentionModel`/`build_multiplane_model` (Kaggle can't `import
src`, same convention as every prior notebook) must add and forward an
`img_size=336` parameter through to `build_backbone()`. 336 is evenly
divisible by DINOv2's patch size 14 (24×24 patch grid, vs. 224's 16×16)
— a clean interpolation target, no partial-patch edge case, but this is
the **first time this project runs DINOv2 at any resolution other than
224**, so the pre-flight smoke test (§3.2) must confirm it actually
works on real data before any real training time is spent, not assumed
safe by the parameter existing.

`select_group()`/`expand_slot_groups()`/`SlotCacheDataset` need no
changes — re-confirmed directly against current code (`src/dataset.py`,
`src/features.py`): all three operate on tensor shape read from the
cache array itself, no literal `224` in any control-flow path (the `224`
that appears in `src/dataset.py` is comment-only, describing the current
cache's shape, not a hardcoded dimension).

**3.2 VRAM must be re-measured, not assumed from `09v1`/`14v1`'s
settings.** `MICRO_BATCH=8`/`ACCUMULATE_STEPS=4` was A2 v2's own
real measured-safe split at 224px/18-pseudo-slots. At 336px each slot
image carries 2.25x the pixels, so the same micro-batch will not
necessarily fit in the same VRAM headroom. The pre-flight smoke test
(overfit 8 real studies, shape/NaN guards — same practice as every prior
training notebook) must include a real VRAM check at the target
micro-batch, shrinking `MICRO_BATCH` (and raising `ACCUMULATE_STEPS`
to compensate, keeping the effective batch size unchanged) if needed,
same empirical-not-assumed approach A2 v1 used to set its own
`batch_studies=32`. Report whatever the real safe split turns out to be
in the notebook's own "Real output" section.

**Otherwise unchanged from `14v1`:** `AdamW`, `lr_backbone=8e-6`,
`lr_head=1e-3`, `weight_decay=0.02`, last 6 transformer blocks unfrozen,
12 epochs, OneCycle, `swa_epochs=3`, a pinned seed (matching `14v1`'s own
correction), `expand_groups=True` (18 pseudo-slots, A2 v2's winning
architecture).

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
gold studies — unchanged from every prior fold-0 pilot).

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
- **Graduating anything to `src/`.** If this pilot (or its eventual
  4-fold scale-up) is adopted, `src/model.py::SlotAttentionModel.__init__`
  needs the same `img_size` forwarding fix identified in §3.1, and
  `src/config.py::SLOT_CACHE_IMG_SIZE` (currently hardcoded `224`) needs
  updating or parameterizing. Real, priced work, deferred to that actual
  graduation decision — not designed here, same discipline as every
  prior pilot in this project.
- **Porting the wide window and 336px resolution into the live-decode
  submission path** (`notebooks/11v1_a2v2_submission_inference.ipynb`) —
  needed only if this graduates to a real submission. `11v1`'s current
  `WINDOW = (0.35, 0.65)` uniform tuple would need converting to a
  per-plane lookup (A3 v2's own §5 already priced this), and its decode
  step would need the 336px crop/resize path added — both real,
  deferred work.
- **RadImageNet pretraining and ROI/spatial-attention pooling** — the two
  other leads considered for the same weak-finding cluster, not chosen
  this round, still on the table if this pilot's result doesn't move the
  needle enough.
- **The `synovitis` label-noise problem** — a distinct, separately
  diagnosed issue (RESOURCES.md, `starkhushi`), not addressed by
  resolution or window changes.

## 6. Testing strategy

No `src/` changes in this spec (§3.1/§5 — the resolution/window changes
are notebook-only until a real graduation decision), so no new unit
tests are owed. The existing suite (90/90 passing as of A2 v2's
graduation) is not expected to change. Validation is entirely real-
Kaggle-output-based, same discipline as every prior item:

1. `15v1`: the two-field fingerprint diff (§2.1) run and reported before
   the probe; the `13v1`-vs-`15v1` `window`-field sanity check;
   `--probe 40`'s measured rate reported against the documented 9.0/s
   baseline (not compared to `13v1`'s own slower number); full build
   reports `complete: N/N shards`; `cache_meta.json` reads `img: 336`,
   `window: "default"`.
2. `16v1`: cache-identity guard and fold-assignment re-assertion before
   training; the pre-flight smoke test (§3.2) confirms DINOv2 actually
   trains at 336px on real data (shape/NaN guards) and reports the real
   safe `MICRO_BATCH`/`ACCUMULATE_STEPS` split before any full-epoch
   training runs.
3. `16v1`'s own "Real output" section filled in with the real per-finding
   table (both best-epoch and SWA readouts), the §4 macro check, the
   directional read including `mcl_injury`'s context delta, and the
   resulting scale-or-stop decision per the pre-agreed rule — computed
   for real, not estimated.

## 7. Process note

Per [[feedback-opus-review-spec-before-implementing]]: this spec gets an
Opus-model review before the implementation plan is written, same
process as every prior architecture spec in this project.
