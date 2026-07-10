# Integrating a trained Tokyo Drift NN model

How a converted PyTorch FPWFS model becomes a usable `model` predictor in
this pipeline. Written after integrating the second model; the steps are
now routine for same-family models. Read this before adding a new one.

## What's already integrated

| mode | filter | modes | run | notes |
|---|---|---|---|---|
| `vampires_f760_10zern` | F760 | 10 | `974j9jqt` (2023-10) | first model; the reference |
| `vampires_f750_35zern` | F750 | 35 | `CHP143~1` (2024-04) | nocoro baseline |

Both are the same **`FFModel` family** (two broadband PSF frames + the
applied DM move → predicted modal wavefront error). Adding another
same-family model is config + a checkpoint, no code.

## The pieces

- **`model_torch.py`** — the PyTorch `FFModelTorch`, vendored *byte-identical*
  (md5 `900824d721e7f24928b14ef5e2d2c29a`) from the conversion project
  (`external/tokyo_drift_training/pytorch_conversion/model_torch.py`). It is
  self-describing: a checkpoint's `meta` carries the full architecture
  (`n_modes`, conv/dense sizes, …), so one class loads every FFModel.
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
