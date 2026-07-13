# Closed-loop timing benchmark

`benchmark.py` answers "where does an iteration's time go?" It runs the
**production closed loop** — built exactly like the `run()` entry
point's sim branch, against the misaligned bench sim — with lightweight
timing wrappers around each seam, then prints a per-stage breakdown.
The loop code itself is untouched: stages are measured by wrapping the
callables `run_closed_loop` already consumes, so the numbers reflect
the real wiring, not a synthetic reconstruction.

## Running it

From the repo root, with any normal run config (the shipped sim ini by
default):

```bash
python -m fpwfsc.tokyo_drift.benchmark                                   # defaults
python -m fpwfsc.tokyo_drift.benchmark --mode vampires_vvc_f750_35zern_crop --error-rms 0.02
python -m fpwfsc.tokyo_drift.benchmark --device mps --iters 20           # or cuda / cuda:0
python -m fpwfsc.tokyo_drift.benchmark my_run.ini --predictor oracle     # checkpoint-free
```

| option | meaning |
|---|---|
| `config` (positional) | run config `.ini`; default = the shipped sim ini |
| `--mode` | override `[MODE] mode name` |
| `--device` | override `[MODEL] device` (`cpu` / `mps` / `cuda` / `cuda:0`) |
| `--iters` | override `[LOOP_SETTINGS] N iter` |
| `--predictor` | `model` / `oracle` / `random_walk` |
| `--error-rms` | override `[SIMULATION] initial error rms` — the ini default (0.15) suits the 10-mode model; use ~0.02–0.05 for the 35-mode modes or the loop wanders out of distribution and trips DM safety |

With no `[MODE] calibration profile` configured, a full calibration is
fitted against the bench first and reported as a one-time setup cost,
so the loop always runs with realistic alignment. The `[SNR]` settings
(frame averaging, sim flux) are honored like any run.

## Reading the report

**One-time setup** — sim construction, calibration, model load, and
*first inference (warmup)*: the first forward pass pays lazy allocation
and, on GPU backends, kernel compilation (~1 s on Metal), so it is
quarantined out of the steady-state means.

**Per-stage** — per-iteration costs, with calls / total / mean / share:

| stage | what it covers |
|---|---|
| `camera: TS2 render` | one optical propagation per exposure (the wavelength-sampled, coronagraph-included telescope-sim render) |
| `camera: mangle + noise draws` | rotate → off-center crop → flux scale, plus the N Poisson+read-noise draws of `frames to average` |
| `frame reduction` | background / flat / bad-pixel handling |
| `preprocess` | rotate → center → crop to the model frame |
| `NN inference` | `predictor.predict` (see device notes) |
| `command translation` | mode coefficients → 50×50 microns command (TranslationDM) |
| `DM write` | shipping the command to the (sim) DM |
| `loop overhead` | everything unaccounted: min-max normalization, integrator math, safety checks |

The modal-residual trajectory is printed last as a sanity check that
the timed run actually converged.

Reference numbers (Apple-silicon laptop, 2026-07; VVC 35-mode `_crop`
mode, easy preset, `average=8`): TS2 render ~370 ms/frame (≈75% of the
543 ms iteration), mangle+noise ~74 ms, NN inference 58 ms on `cpu` /
36 ms on `mps`, everything else ≤1 ms. Setup: sims ~5 s, calibration
~12 s, model load ~0.5 s.

**Interpreting sim vs hardware:** in sim, the TS2 render dominates —
that column *is* the simulator, not the pipeline. On real hardware the
render is replaced by camera exposure + readout, and the largest
software terms in the loop-latency budget become NN inference and
preprocessing. That is what the `--device` knob is for sizing.

## Device notes (CPU / MPS / CUDA)

The inference path is device-generic and does the placement explicitly
at every step, so a CUDA GPU needs **no code changes** — set
`[MODEL] device` (the GUI field) or `--device` to `cuda` (or `cuda:0`,
any torch device string):

- weights move once at load (`model.to(device)` in `TorchPredictor`);
- each call's inputs are created directly on the device
  (`torch.as_tensor(..., device=device)` in
  `FFModelTorch.predict_nhwc`);
- outputs return via `.cpu().numpy()`.

MPS is *not* a special case that bypassed this: despite Apple's unified
memory, PyTorch treats `mps` as a distinct device, and a missing
placement raises the usual all-tensors-must-be-on-one-device error —
so the validated MPS path exercises exactly the transfers CUDA needs.
The `.cpu()` on the output also forces device synchronization, which
makes the benchmark's inference wall-times honest on any GPU (no async
launches escaping the timer).

Caveats: the device string is passed to torch unvalidated — a typo or
an unavailable backend raises at model load, not silently. Inference
runs float32 everywhere (the checkpoint's native precision).

## Gotchas

- Prefix `KMP_DUPLICATE_LIB_OK=TRUE` when invoking from a shell where
  hcipy may load before the predictor module (the package sets it
  defensively, but import order can bypass that under pytest/tools).
- On real VAMPIRES hardware, `average != 1` currently means a hardcoded
  50-frame mean (upstream `Vampires.take_image` quirk — see
  `PENDING_VAMPIRES_INTEGRATION_NOTES.md`), so hardware timings of the
  camera stage will reflect 50 frames until that is fixed.
