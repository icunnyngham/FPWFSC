# tokyo_drift — NN focal-plane wavefront sensing & control

Closed-loop focal-plane wavefront control driven by a neural network.
Like Fast & Furious, the method uses temporal diversity: the NN takes
the last **two** focal-plane frames plus the DM actuation applied
between them and predicts the current wavefront error, which is then
corrected with a leaky integrator. The two-frames-plus-actuation input
disambiguates modes that are degenerate in a single image.

The simulator backend is [telescope-sim](https://github.com/icunnyngham/telescope-sim)
(TS2): the same optical configuration used to train a model drives the
in-loop simulation, and a deliberately misaligned "bench sim" derived
from it stands in for real hardware so that camera/DM alignment
calibration can be exercised end-to-end without instrument time.

**Status: functionally complete in sim mode.** Entry point, GUI,
simulator backends, calibration workbench, session logging, and the
trained-NN inference path are all in place. Real-hardware backends are
the remaining work.

## Install

```
pip install -e .[tokyo-drift]    # adds telescope-sim and torch
```

## Run

```
python -m fpwfsc.tokyo_drift.run          # headless, sim mode defaults
```

Like every FPWFSC pipeline, a run is fully described by its `.ini`
file (validated against `tokyo_drift_config.spec`); the GUI, CLI, and
notebooks all call the same `run(camera, aosystem, config, configspec,
my_event, plotter)`.

## Calibration

Alignment calibration is a separate, explicit step — the loop only
ever consumes a saved profile (`calibrations/<name>.yaml`, selectable
in the GUI or named in the `.ini` for scripted runs; a path to a YAML
also works).

In the GUI: **Calibrate** runs the staged fit against the simulated
bench, streaming each stage (probe frame -> rotation -> center ->
flips -> scale) to the alignment panels with the sweep-score curves;
the fitted values land in the collapsible **Calibration parameters**
panel, where hand-edits re-render the panels live from the cached
probe frame; **Save calibration** stores a named profile and adds it
to the dropdown.

Headless:

```python
from fpwfsc.tokyo_drift.calibration import calibrate_bench_sim, save_profile
profile, report = calibrate_bench_sim("vampires_f760_10zern",
                                      preset="easy", seed=27)
save_profile("fitted_easy_27", profile)
```

Calibration fits on a known asymmetric probe poke (coma + trefoil):
a flat-wavefront PSF is centro-symmetric, so rotation and flips are
not identifiable from a null frame.

## Predictors

`[LOOP_SETTINGS] predictor` selects what drives the loop:

- **`model`** — the trained NN. Loaded from the mode's checkpoint (the
  manifest `checkpoint:` pointer, resolved relative to the mode dir).
  Checkpoints are large and **never committed** (`.gitignore`); drop the
  converted `.pt` into the mode directory. The `.pt` is self-describing
  (its `meta` carries the full architecture), so nothing model-specific
  is hardcoded. Per-inference latency is printed once at loop start.
- **`oracle`** — sim-only; reads the injected truth through a side
  channel. The loop must converge; validates everything except the NN.
- **`random_walk`** — pure noise; the loop must diverge (a sanity check
  that no information is leaking).

The NN model wrapper (`model_torch.py`) is vendored byte-identical from
the validated conversion project (the same code the closed-loop eval
bench proved framework- and sim-equivalent). Two sign conventions bridge
the loop and the model, both pinned by tests against the training-matched
ideal sim (where the model is near-perfect):

- the prediction is the **current** wavefront error; the loop applies
  `state = leak*state - gain*prediction` (no negation in the predictor);
- the loop reports the actuation as a DM **command** delta
  (`state_k - state_{k-1}`); the model was trained on the **aberration**
  delta (`before - after`), so the adapter feeds it `-delta_actuation`.

Because the NN needs temporal diversity to disambiguate sign-degenerate
modes, the loop applies a small known **initial diversity move**
(`[MODEL] initial move sigma`) before the first prediction — matching the
training-time eval loop. The dummy predictors ignore actuation and run
with no initial move.

> **torch + hcipy / OpenMP:** both link an OpenMP runtime; loading torch
> alongside hcipy aborts with a libomp double-init unless
> `KMP_DUPLICATE_LIB_OK=TRUE` is set. The model predictor sets it
> defensively before importing torch, so in-process CPU inference in the
> `telescope-sim-dev` env just works.

## Config sections

| Section | Purpose |
|---|---|
| `[MODE]` | Which trained-model mode to run (a telescope-sim config + NN checkpoint pair under `modes/`), and which saved calibration profile to apply |
| `[LOOP_SETTINGS]` | Iterations, gain, leak factor, predictor selection (`model` / `oracle` / `random_walk`), Strehl estimator (`vandam` / `proxy`), optional Strehl early-stop |
| `[MODEL]` | NN inference: initial diversity-move RMS, torch device (`cpu` / `mps` / `cuda`) — used only by the `model` predictor |
| `[DM]` | MILK shared-memory channel and command safety bounds |
| `[SIMULATION]` | Bench-sim misalignment preset, seed, injected initial error (sim mode only) |
| `[CAMERA CALIBRATION]` | Background / masterflat / bad-pixel FITS files (empty fields fall back to border-median background estimation) |
| `[IO]` | Per-iteration session logging; hitchhiker file-stream mode |

## Session logs

With `save_log = True`, every run writes a timestamped directory under
`log_path`:

```
tokyo_drift_<timestamp>/
├── config.json          validated settings snapshot
├── iter_NNN/            raw.fits, processed.fits, dm_command.fits,
│                        metadata.json (strehl, state, prediction, rms)
└── summary.json         strehl history, final state
```

Strehl is measured with the van Dam estimator (sub-pixel peak +
aperture photometry) against a pristine-system reference frame by
default; `strehl method = proxy` selects the cheaper peak-to-total
flux ratio.

## Tests

```
pytest fpwfsc/tokyo_drift/tests
```
