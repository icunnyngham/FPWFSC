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

**Status: under construction.** The entry point and config contract are
in place; simulator backends, the GUI, calibration tooling, and NN
inference land incrementally.

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

## Config sections

| Section | Purpose |
|---|---|
| `[MODE]` | Which trained-model mode to run (a telescope-sim config + NN checkpoint pair under `modes/`), and which saved calibration profile to apply |
| `[LOOP_SETTINGS]` | Iterations, gain, leak factor, optional Strehl early-stop |
| `[DM]` | MILK shared-memory channel and command safety bounds |
| `[SIMULATION]` | Bench-sim misalignment preset and seed (sim mode only) |
| `[IO]` | Per-iteration logging |

## Tests

```
pytest fpwfsc/tokyo_drift/tests
```
