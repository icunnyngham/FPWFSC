# Tokyo Drift — quickstart (branch `tokyo_drift` of icunnyngham/FPWFSC)

Tokyo Drift is a neural-network focal-plane WFS&C pipeline for FPWFSC —
an NN analogue of Fast & Furious: two consecutive focal-plane frames plus
the DM move between them → predicted modal wavefront error → leaky-
integrator correction. It ships fully working in **simulation mode**
(training-matched simulator + a deliberately misaligned "bench" simulator
+ an automated calibration workbench); the real-hardware backend
(SCExAO/VAMPIRES) is stubbed pending fixes to `common/bench_hardware.py`
(audited in `fpwfsc/tokyo_drift/PENDING_VAMPIRES_INTEGRATION_NOTES.md`).

Everything lives in one new subpackage, `fpwfsc/tokyo_drift/`; the only
change outside it is the `[tokyo-drift]` optional-dependency block in
`pyproject.toml`.

## 1. Get the code

```bash
git clone https://github.com/icunnyngham/FPWFSC.git
cd FPWFSC
git checkout tokyo_drift
```

(Or add `icunnyngham` as a remote to an existing FPWFSC clone and check
out `tokyo_drift` from there.)

## 2. Install

Python 3.10–3.12 recommended (developed on 3.11). From the repo root:

```bash
pip install -e ".[tokyo-drift]"
```

The extra adds `telescope-sim>=2.1` (the optical simulator, on PyPI),
`torch` (CPU is fine — inference is ~60 ms/frame), and `pyyaml` on top
of FPWFSC's base dependencies.

## 3. Unpack the model checkpoints

The trained weights are too large for the repo (gitignored). Place
`tokyo_drift_checkpoints_26-07-12.tar` (~1.5 GB) at the **repo root**
and:

```bash
tar -xf tokyo_drift_checkpoints_26-07-12.tar
```

That drops three checkpoints into `fpwfsc/tokyo_drift/checkpoints/`:

| mode | instrument / filter | checkpoint |
|---|---|---|
| `vampires_f760_10zern`          | VAMPIRES F760, 10 Zernike modes        | `974j9jqt_torch.pt` |
| `vampires_f750_35zern`          | VAMPIRES F750, 35 Zernike modes        | `CHP143_torch.pt`   |
| `vampires_vvc_f750_35zern_crop` | VAMPIRES F750 + vector-vortex coronagraph (charge 4), 35 modes | `CKP8EJ_torch.pt` |

Sanity check:

```bash
python -c "from fpwfsc.tokyo_drift.mode_registry import list_modes, checkpoint_path; \
  [print(m, '->', checkpoint_path(m).name) for m in list_modes()]"
```

(Without the checkpoints everything still runs — the GUI flags the NN as
unavailable and the debug predictors still work — but the interesting
predictor is obviously the model.)

## 4. Run the GUI

```bash
python fpwfsc/tokyo_drift/tokyo_drift_GUI.py
```

Suggested first session (all defaults are sane):

1. **Instrument**: `Sim`. **Mode**: start with `vampires_f760_10zern`.
2. **Calibrate** — the "Model↔instrument alignment" panel. Click
   **Auto-calibrate**: it pokes a probe pattern on the simulated
   (misaligned) bench and fits camera rotation, PSF center, flips, and
   the DM actuation scale, streaming each stage to the plot. Save the
   profile when it finishes and select it in the profile dropdown.
   (The simulated bench draws its misalignments from the `bench sim
   preset` prior — the calibration is fitted blind and scored against
   the injected truth, which the GUI reports.)
3. **Run** — the closed loop injects a hidden wavefront error and
   corrects it with the NN. For the non-coronagraph modes, watch the
   Strehl panel converge (→ ~0.95+ in a handful of iterations).
   Each run injects a fresh random error against the same bench, so
   Run can be clicked repeatedly to see different episodes.
4. **The coronagraph mode** (`vampires_vvc_f750_35zern_crop`) works the
   same way — calibrate, then run — but read the mode-coefficient panel
   rather than Strehl for convergence (behind a coronagraph the Strehl
   proxy only tracks leakage). Its speckle field is faint, so it relies
   on the "Noise mitigation" settings; the defaults (8-frame averaging)
   are already set for it.

## 5. Tests

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest fpwfsc/tokyo_drift/tests -q
```

134 tests, ~2.5 min (a few run full calibrations and closed loops).
Model-dependent tests skip automatically if checkpoints are absent. The
env var works around the torch+hcipy OpenMP double-load on some
platforms (the package sets it internally for normal use; pytest import
order can bypass that).

## 6. Where to read more

- `fpwfsc/tokyo_drift/README.md` — architecture + how the pieces fit.
- `fpwfsc/tokyo_drift/MODEL_INTEGRATION_NOTES.md` — how trained models
  integrate, the loop↔model conventions, and the coronagraph specifics.
- `fpwfsc/tokyo_drift/checkpoints/README.md` — checkpoint provenance.
