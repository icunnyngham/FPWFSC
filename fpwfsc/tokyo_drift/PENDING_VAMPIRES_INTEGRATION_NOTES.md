# Pending VAMPIRES hardware integration — audit notes

tokyo_drift is functionally complete in sim mode; the real-hardware
backend is the remaining work (`run()` currently raises
`NotImplementedError` for non-Sim). Before wiring it, we audited the
existing VAMPIRES connect path in this repo against the last
known-working deployment — the Sept 2024 FnF snapshot that ran at
SCExAO (`/home/scexao/Documents/FnF`). Summary: tokyo_drift's sim/real
boundary already matches the deployed contract end to end, but the
shared `common/bench_hardware.py` has bit-rotted since the Sept 2024
port and currently blocks *any* pipeline (fnf included) from
instantiating VAMPIRES hardware.

## What already matches

- **Instrument selection** — `gui_helper.load_instruments('Vampires')`
  returns `hw.Vampires(), hw.SCEXAO()`, the same pair fnf uses. The
  import is lazy so the GUI works off-instrument (see "suspected
  broken" below for why fnf's eager import does not).
- **Camera contract** — the loop consumes `take_image(average=)`
  exactly as `hw.Vampires` provides it (`vcam1` shm stream).
- **DM contract** — the loop ships an absolute 50×50 command in
  **microns of surface, float32** and ignores the return value; that
  is precisely what `hw.SCEXAO.set_dm_data` writes to the DM shm
  channel. Verified against the 2024 bench sessions: `surf * 1e6 →
  float32 → shm.set_data(...)` plus the same 10 ms settle.
- **Modal → command translation** — `dm/translation_dm.py` is a
  faithful port of the bench-validated `SubaruZernikeDM` (HCIPy DM
  built directly on the 50×50 actuator grid; `dm_actuate_scale`
  carries the meters-per-coefficient calibration, ~1.4e-6 on the
  bench). It replaces `SCEXAO.make_dm_command` by design — same units
  to the same stream, but from mode coefficients rather than a
  resampled phase map.
- **Frame reduction** — `sf.equalize_image` with the standard
  `[CAMERA CALIBRATION]` files and border-median fallback, same as the
  other pipelines.
- **Preprocessing** — tokyo_drift keeps its own rotate → center →
  crop → flip chain instead of `sf.reduce_images`, deliberately: at
  VAMPIRES' large camera rotation (~235°) alignment must happen after
  rotation; `reduce_images` aligns first and is only valid for small
  angles.

## Suspected broken in the current repo (predates tokyo_drift)

These affect fnf's VAMPIRES path too, and want fixing in `common/`
rather than in this pipeline:

1. **`common/bench_hardware.py` is not importable off-Keck.** The
   Keck-only imports (`aoscripts`, `aosys`, lines ~8-10) sit *outside*
   the try/except meant to guard them, and `import ipdb` precedes
   them. Confirmed: `import fpwfsc.fnf.gui_helper` fails in a clean
   env (fnf's gui_helper imports bench_hardware eagerly at module
   top, so the fnf GUI cannot even launch).
2. **The Subaru classes reference names the module never imports** —
   `shm` (pyMilk), `filters` (vampires_control), `time`, `sf`, `pf`.
   The Sept 2024 `hardware.py` had these imports; they were dropped
   when the classes were ported into `bench_hardware.py`. So even on
   the SCExAO machine, `hw.Vampires()` / `hw.SCEXAO()` would
   `NameError` today.
3. **fnf's hardware branch has drifted Keck-ward** and would crash
   with the Subaru classes: `run.py` calls
   `AOsystem.AO.get_cog_filename()` (SCEXAO has no `.AO`), unpacks a
   tuple from `set_dm_data` (SCEXAO returns None), and never calls
   `make_dm_command`. The fnf GUI also passes Keck-shaped `aoargs`
   (`rotation_angle_dm`, `flip_x`, `flip_y`) that `SCEXAO.__init__()`
   does not accept. The Sept 2024 `run_bench.py`, not the current fnf
   `run.py`, is the working reference for the Subaru flow.
4. **`SCEXAO` hardcodes `dm00disp04` with a no-arg constructor.** The
   bench sessions used different channels at different times
   (`dm00disp02` in May 2024), so the channel should be a constructor
   parameter. tokyo_drift already carries a `[DM] dm channel` config
   field to plumb into it.

## Remaining work for the tokyo_drift hardware backend

The sim boundary (`take_image(average=)` / `set_dm_data(50×50 µm)`)
was designed to be the hardware boundary, so the adapter itself is
thin. Beyond the `common/` fixes above:

- **Filter assertion** — tokyo_drift locks wavelength/pixel scale to
  the trained mode instead of reading camera keywords (necessarily —
  the NN is filter-specific). The hardware branch should assert
  `Vampires.filter_name` against the mode manifest's `filter:` at
  connect time.
- **Pixel scale** — `hw.Vampires` hardcodes 5.9 mas/pix (flagged
  "need to confirm"); the modes were trained at 6.0. If 5.9 is real,
  that ~1.7% sampling mismatch is a model error the calibration axes
  (rotation / center / flips / dm_scale) do not absorb. Needs
  confirmation.
- **Frame averaging** — `Vampires.take_image` treats any
  `average != 1` as a hardcoded 50-frame mean; the class should honor
  the requested count (upstream `common/` fix). This now matters more:
  the loop-averaging option exists (`[SNR] frames to average`, restoring
  the Sept 2024 deployment's `N images averaged`), the calibration
  harness requests `average=16`, and the coronagraph modes *require*
  averaging (or more flux) — single noisy coro frames diverge the NN
  loop (see MODEL_INTEGRATION_NOTES, coro section).
- **Darks** — the bench sessions subtracted a `vcam1_dark` shm frame;
  `hw.Vampires` does no dark handling. Covered by pointing
  `[CAMERA CALIBRATION] background file` at a FITS snapshot of that
  stream; document as the bench procedure.
