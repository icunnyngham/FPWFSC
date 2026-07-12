# Integrating a trained Tokyo Drift NN model

How a converted PyTorch FPWFS model becomes a usable `model` predictor in
this pipeline. Written after integrating the second model; the steps are
now routine for same-family models. Read this before adding a new one.

## What's already integrated

| mode | filter | modes | optics | run | notes |
|---|---|---|---|---|---|
| `vampires_f760_10zern` | F760 | 10 | no coro | `974j9jqt` (2023-10) | first model; the reference |
| `vampires_f750_35zern` | F750 | 35 | no coro | `CHP143~1` (2024-04) | nocoro baseline |
| `vampires_vvc_f750_35zern_crop` | F750 | 35 | VVC charge-4, 120px | `CKP8EJ~6` (2024-06) | randcrop → 120px input |

(A 128px VVC model, `CB9WJU~4`, was integrated then **dropped** — it
diverges in closed loop even on its own ideal Zernike-DM sim, at every gain,
while `CKP8EJ~6` converges there cleanly. A model-specific issue, not a
config one; re-add if it's understood. See the coro section.)

All three are the same **`FFModel` family** (two broadband PSF frames + the
applied DM move → predicted modal wavefront error). Adding another
same-family model is config + a checkpoint, no code — even the coronagraph
lives entirely in the TS2 sim config, not the network (see the coro section
below for the extra config it needs).

## The pieces

- **`model_torch.py`** — the PyTorch `FFModelTorch`, vendored *byte-identical*
  (md5 `17174a7aab59f033e342c15dc6b4fe39`) from the conversion project
  (`external/tokyo_drift_training/pytorch_conversion/model_torch.py`). It is
  self-describing: a checkpoint's `meta` carries the full architecture
  (`n_modes`, conv/dense sizes, `input_hw`, …), so one class loads every
  FFModel — including the 120px crop model (`input_hw=120`). Re-vendor if the
  upstream copy changes; `input_hw` defaults to 128 so older checkpoints are
  unaffected.
- **`model_predictor.py::TorchPredictor`** — wraps it as a loop predictor
  (`predict(frames, actuation) -> ndarray`). `TorchPredictor.from_mode(name)`
  resolves the checkpoint and builds the model from `meta`.
- **`modes/<mode>/`** — `ts2_config.yaml` (the optical config the NN was
  trained against) + `manifest.yaml` (metadata + `checkpoint:` pointer).
- **`checkpoints/<mode>/<file>.pt`** — the weights (gitignored; see
  `checkpoints/README.md`).

## The seam contract (how the loop and model connect)

Both conventions below are pinned by tests in `tests/test_model_predictor.py`
against the training-matched ideal sim (where a correct model is near
perfect); get them wrong and the loop diverges or stalls.

- **Frames**: the loop hands `np.stack([prev, cur])` (2, 128, 128), each a
  min-max-normalized `[0,1]` PSF (`loop._normalize`, identical to TS2's
  `per_sample_norm`). The model wants `concat([before, after])` → (128,128,2);
  `predict_residual(psf0=before, psf1=after, …)` does that.
- **Return sign**: the prediction is the **current** wavefront error; the
  loop applies `state = leak*state - gain*prediction`. Do **not** negate in
  the predictor.
- **Actuation sign** (the subtle one): the loop reports `delta_actuation` as
  the DM **command** change between the two frames (`state_k - state_{k-1}`).
  The model was trained on the **aberration** change (`before - after`).
  Net wavefront = external error + command, so these are opposite → the
  adapter feeds the model `-delta_actuation`. Verified decisively on the
  35-mode model: correct sign → cos 0.94 with truth, flipped → cos -0.09.
- **Initial diversity move**: the NN needs a known move between the first
  two frames to disambiguate sign-degenerate modes. `run.py` draws one
  (`[MODEL] initial move sigma`, default 0.01) and passes it to
  `run_closed_loop(initial_move=…)`; the dummy predictors pass `None`.

## Adding a new (same-family) model — step by step

1. **Find the converted checkpoint + its eval bundle** in the model-side
   project: `external/tokyo_drift_training/pytorch_conversion/<id>/<name>.pt`
   and `external/tokyo_drift_training/eval/runs/<id>/`. The bundle's
   `run.yaml` + `ts2_*.yaml` are the source of truth for the TS2 config and
   `n_modes`; its README records the conversion parity.

2. **Confirm it's the FFModel family** — check `meta['arch']` is
   `FFModel multi_output` and `meta` has the arch keys. If it's a different
   family (coronagraph, extra inputs, single output), this is **not** a
   drop-in: it needs a new I/O contract (mirror the model-side
   `eval/core/families.py`) and possibly a sampler-variant adapter. Escalate
   to the model-side handoff before proceeding.

   ```python
   import torch; print(torch.load("<name>.pt", map_location="cpu")["meta"])
   ```

3. **Create `modes/<mode>/ts2_config.yaml`** — copy an existing mode's config
   and change only what the eval bundle's TS2 config changes (for F750/35z
   that was three fields: `central_lam`, `fractional_bandwidth`,
   `zernike_dm.n_modes`). Keep `module: fpwfsc.tokyo_drift.miles_pupil` (the
   vendored pupil), not the eval bundle's CWD-relative helper path. **This
   config parity is the #1 integration risk** — any pupil / focal-grid /
   filter / normalization / mode-count difference puts the model
   out-of-distribution and degrades silently.

4. **Create `modes/<mode>/manifest.yaml`** — `mode`, `instrument`, `filter`,
   `basis`, `ts2_config: ts2_config.yaml`, `checkpoint: <file>.pt`, `notes`.

5. **Install the weights** at `checkpoints/<mode>/<file>.pt` (symlink into
   the training project locally; see `checkpoints/README.md`). Update that
   README's provenance table.

6. **Verify** (before writing test thresholds — measure first):
   ```python
   from fpwfsc.tokyo_drift.mode_registry import checkpoint_path, mode_n_modes
   from fpwfsc.tokyo_drift.model_predictor import TorchPredictor
   checkpoint_path("<mode>"); mode_n_modes("<mode>")
   TorchPredictor.from_mode("<mode>")            # n_modes matches meta
   ```
   Then the ideal-sim sign check and an easy-preset bench loop (see the
   snippets used in this session / the tests). Expect the ideal-sim cos to
   be high and the flipped sign to collapse; expect the bench loop to
   converge to Strehl ≳ 0.9 at a *training-appropriate* error level.

7. **Add it to the tests** — append a dict to `MODES` in
   `tests/test_model_predictor.py` (`name`, `n_modes`, `err_rms`, `min_cos`);
   the sign + convergence tests parametrize over it and skip if the
   checkpoint is absent. Add a row to the `test_mode_registry` parametrize.

8. **GUI** — no changes. The mode appears in the dropdown automatically; the
   NN-availability alert and checkpoint resolution already handle it.

## Coronagraph (VVC) models — extra config, still no net changes

The VVC models proved the coronagraph is a *sim-config* concern: same
FFModel, the coronagraph is just added optics. What their `ts2_config.yaml`
needs beyond the no-coro template (all from the eval bundle, which is the
source of truth):

- **A `coronagraph:` block** — `type: vector_vortex`, `charge: 4`, and a
  `lyot:` sub-config (an `external_pupil` with its own kwargs). TS2 renders
  this natively (the eval bundle validated TS1↔TS2 coro parity).
- **`miles_synthpsf` aperture, not `miles_pupil`** — the coro era used a
  different pupil generator (it has a `spider_scale` arg the Lyot needs, so
  `miles_pupil` can't stand in). Vendored byte-identical at
  `miles_synthpsf.py` (md5 `f6c39d2e92955d2560d25b66f1bbf7b4`; all upstream
  copies are identical — see the model-side `DIFFS.md`). Both the aperture
  and the Lyot reference it, with different kwargs.
- **Auto-derived `zernike_diameter`** — the legacy coro sampler derives the
  Zernike DM diameter at runtime (`1.01 × max pairwise aperture distance`),
  so the fixture's `7.79` is a *dead key* that mismatches by ~8% at nonzero
  actuation. Use the per-pupil literal from the eval bundle (7.9430470876 at
  pupil res 128, 7.9053295407 at res 256). This is the easiest thing to get
  silently wrong.
- **120px crop variant**: the `_crop` model trained on a 120px centered crop.
  A 120px focal plane at the same 6 mas/pix (`focal_res: 120`,
  `focal_extent: 0.720`) is bit-identical to that crop, so just render 120px
  — the whole pipeline (ideal sim, preprocess crop, model `input_hw=120`)
  follows the focal size with no crop code.

### Coro closed loop on the bench — diagnosed and (mostly) fixed (2026-07-11)

The coro models validate on the ideal sim (correct sign) but **diverged in
closed loop on the actuator-grid bench**, even at 0.001 rms. A first
investigation blamed a fundamental actuator-grid-vs-Zernike wavefront
mismatch; a same-day follow-up **overturned that** with controlled
experiments. Two independent causes, neither fundamental:

1. **Single-frame noise** (the actual divergence driver). `run()` calls
   `run_closed_loop` with `average=1`; a coro frame is faint speckle, and
   photon+read noise on a single min-max-normalized frame corrupts the NN
   input enough to diverge the loop *even with a perfect calibration
   profile*. `average>=4-8` fixes it (noiseless → 0.016 residual;
   average=4 → 0.019; average=1 → diverges to 0.30+). This is why every
   earlier dm_scale sweep "still diverged", and why smaller injected
   errors diverge *harder* (dimmer speckle signal, same noise). The
   earlier "noise ruled out" was a single-step-cosine test — it doesn't
   capture the closed-loop compounding. The bright no-coro core has the
   SNR to tolerate average=1. **Fixed via the `[SNR]` config section**
   ("Noise mitigation" in the GUI): `frames to average` passes straight
   to `take_image(average=N)` (in sim: N noise draws of ONE rendered
   frame, nearly free), and `int phot flux exponent` (sim-only, 10^x
   photons/m^2, legacy notation; training-era default 3.5) sets the
   bench-sim source brightness — measured coro thresholds on the easy
   bench: 10^4.0 converges at single frames, or average>=4-8 at 10^3.5.
   The coro mode now has its own modal-residual bench-convergence test.
2. **Command amplitude: the effective DM gain is ~1.6-1.7, and the old
   calibration missed it unreliably.** The influence functions render a
   smooth commanded Zernike surface ~1.6x larger than the commanded poke
   amplitudes (per-mode 1.43-2.31 across 35 modes; a property of the
   nominal DM, same physics as the real bench's empirical 1.4e-6 vs 1e-6
   nominal `dm_actuate_scale`). At the *correct* scalar, bench-vs-ideal
   frames at the same modal state match at cosine 0.998+ **even through
   the VVC** (the ~12% high-order influence-function print-through is
   invisible in the PSF, and a full 35x35 modal-inverse correction buys
   nothing over the scalar). Calibration used to compare bench pokes
   against ideal references rendered at the *commanded* amplitude — 1.6x
   apart in aberration strength, where coro frames decorrelate to cosine
   ~0.3 — degrading the rotation/center fits and biasing `fit_dm_scale`
   toward ~1.0 on some seeds. **Fixed (gain-aware calibration)**:
   `calibration/dm_gain.py::nominal_dm_gain` precomputes the nominal
   overshoot from the DM model (spec-sheet knowledge, not injected
   truth); `acquire_probe` renders every reference at the gain-matched
   amplitude and the `dm_scale` sweep centers on the nominal gain. Coro
   recovery went from bimodal (scale fits of ~1.0 or ~1.6, rotation
   errors 3.6-5.6°) to tight (over-truth 1.66-1.74, rotation 0.2-2.7°
   across seeds).

With both in place (fitted profile + `average=8` wired manually), the
`_crop` coro model **converges on the mangled easy bench**: 0.104 →
0.013-0.018 across seeds, matching the ideal Zernike-DM bench floor.
Related knock-on understanding:

- With gain-matched references, the rotation fit locks onto the probe
  response's full orientation and **partially absorbs the (v1-unfitted)
  DM rotation** into image rotation — the camera-to-DM alignment the loop
  wants, but it inflates the report's pure-image-rotation "error" by up
  to |dm_rot| (see the realistic-preset test).
- The border-median background fallback in frame reduction subtracts
  real coro halo flux and adds stripe noise on lightly averaged frames —
  now **off by default** (`estimate background from border`).
- Don't read the coro loop's Strehl regardless — it's a leakage proxy;
  pupil RMS / the true modal residual is the convergence signal.
- A Zernike-DM ideal bench (no actuator grid, no mangling) also converges
  (0.093 → 0.010) and remains a useful in-distribution diagnostic tier,
  but is no longer needed as a workaround.

## Gotchas (learned the hard way)

- **OpenMP double-load**: torch + hcipy both link libomp → abort. The
  predictor sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch, so
  in-process CPU inference in `telescope-sim-dev` just works.
- **`initial error rms` is mode-dependent.** The model is only in
  distribution up to roughly its training `initial_error_sigma` (0.05 for
  F760, 0.03 for F750). The sim ini default (0.15) is fine for the 10-mode
  model (it walks a 3× error down over a few steps) but is ~5× the 35-mode
  model's scale — deep OOD, the loop wanders and can trip DM safety. Set a
  training-appropriate error for a new model; the per-mode value the tests
  use is in `MODES`.
- **Higher mode counts fit less tightly.** 35 modes reach ideal-sim cos
  ~0.94 and bench Strehl ~0.96–0.99, vs the 10-mode model's ~1.0. Expected
  (the eval bench notes the same), not a bug.
- **Config parity is silent when wrong** — it degrades inference, never
  crashes. Diff the new `ts2_config.yaml` against the eval bundle's TS2
  config field by field.
- **GUI imports must be absolute** (`from fpwfsc.tokyo_drift…`), never
  dotted-relative — the GUI is launched as a script.

## Pointers

- Model-side conversion + closed-loop validation:
  `external/tokyo_drift_training/` (start at `eval/README.md` and the
  `MODEL_SIDE_FPWFSC_INTEGRATION_HANDOFF.md`).
- This pipeline's build history + findings: `../../../tokyo_drift_plan.md`
  (lab-repo root), M8 section and after.
- Checkpoint layout / provenance: `checkpoints/README.md`.
