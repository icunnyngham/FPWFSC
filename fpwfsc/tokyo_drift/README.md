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
