# Pending VAMPIRES hardware integration — audit notes

> **STATUS 2026-08-11: the hardware backend is now implemented** and
> the `common/` breakage below is fixed on this branch — see the dated
> update section at the bottom for what changed, what was verified
> against the Subaru sc6 reference checkout, and the bench-day risk
> register. The audit text below is kept as written (2026-07) for the
> record.

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

---

# 2026-08-11 update — Subaru sc6 reference comparison + backend landed

We obtained `seb_sc6_fpwfsc/` (project root, outside this repo): a
snapshot of the FPWFSC checkout on the Subaru sc6 machine — upstream
`mb2448/FPWFSC` main @ `aa9c5d1` (2025-05-19) plus **uncommitted**
hand-edits that ran F&F on **Palila** at the telescope (`git diff`
inside that clone is the flight-proven delta). Verification of the
audit against it, and what landed on this branch:

## Audit items, resolved

1. **Off-instrument importability** — fixed here the same way the sc6
   edit did: Keck imports guarded in a `try`, plus the SUBARU import
   block (`pyMilk` shm, `sf`, `vampires_control`, `time`).
2. **Missing Subaru imports** — same fix, confirmed against sc6.
3. **fnf drifted Keck-ward** — *deliberately not fixed*: fnf is
   upstream's problem; this branch only makes `common/` importable and
   the Subaru classes constructible. `SCEXAO.__init__` now accepts
   `(dm_channel, rotation_angle_dm, flip_x, flip_y)` and
   `make_dm_command` honors the flips (matching sc6), but our fnf
   `run.py` remains upstream's Keck-shaped version.
4. **`dm00disp04` hardcoded** — fixed: `SCEXAO(dm_channel=...)`,
   plumbed from `[DM] dm channel` (GUI passes it on hardware connect).
   sc6 did NOT fix this; default stays `dm00disp04` = what their
   deployment writes to.

## Deliberate divergence from the sc6 reference

- **`SCEXAO.set_dm_data` stays RAW** (50x50 microns-of-surface float32
  straight to the shm channel). The sc6 uncommitted edit repurposed
  `set_dm_data(phase)` to run `make_dm_command` internally (fnf
  convenience). tokyo_drift's whole contract — TranslationDM, the
  bench sim's TS2 actuator DM, the 2024 bench sessions — is built on
  the raw write, and the truly flight-proven layer
  (`shm.set_data(float32) + 10 ms settle`) is identical either way.
  If anyone diffs the two trees on the mountain, this is why.

## Verified against sc6 / adopted

- **DM pupil geometry**: their `make_dm_command` pastes a 44-actuator
  patch centered at (24, 23) — numerically identical footprint to the
  bench notebooks' `shift_x=1, shift_y=2` on a centered pattern. Their
  knowledge and Ian's manual 2024 derivation agree exactly.
- **Command-aperture taper** (new): TranslationDM now crops commands
  to a square `command_aperture_act` box (default 44, config
  `[DM] command aperture actuators`, 0 disables) applied before the
  shifts — so with profile shifts (1, 2) the footprint lands exactly on
  their paste box. Edge actuators outside the illuminated pupil are
  never commanded. Note the trained basis is 7.79 m (~45.8 act), so the
  44-box trims its outer sliver: sim oracle ceiling drops ~0.95→~0.94
  (tests updated); NN loop unaffected in practice.
- **pyMilk API (RESOLVED, was a "risk")**: modern pyMilk (HEAD
  2026-03) changed `get_data` to `(check, timeout, copy, ...)` —
  `reform`/`sleepT` are gone, so the old xaosim-style
  `get_data(True, True, timeout=1.)` **raises TypeError**. This is why
  the sc6 Palila edits use bare `get_data()`. Our `Vampires.take_image`
  now uses `get_data(check=True, timeout=1.)` per frame and
  `multi_recv_data(N, output_as_cube=True)` for averaging (honoring the
  requested N — the 50-frame hardcode is gone; note sc6's Palila still
  hardcodes 150).
- **Dark handling**: `Vampires.fetch_dark()` reads a `vcam1_dark` shm
  frame (sc6's Palila pattern, minus the silent-zeros fallback);
  `take_image` stays RAW and the dark feeds the pipeline's frame
  reducer (single subtraction point). GUI: auto-fetch on hardware
  connect, status row (green/amber), Refetch button.
- **Filter assertion**: `run.assert_camera_matches_mode` compares the
  camera `FILTER01` keyword against the mode manifest's `filter:` by
  leading number ('F750' ~ '750-50'); hard-fails on mismatch, warns if
  the camera exposes no filter name.
- **Calibration on hardware**: the GUI's View / Auto-calibrate / Fine
  tune now work against the real instrument via
  `calibration.harness.HardwareBench` (raw commands out, dark-subtracted
  frames back — the same two calls the loop makes). No `.truth` on
  hardware, so the harness returns the profile without the sim recovery
  report; judge by the stage previews and then loop convergence.
  NOTE: unlike preflight, calibration sends real DM probe pokes.

## Bench-day interface risk register (VAMPIRES unverified since 2024)

The sc6 reference ran **Palila**, so no VAMPIRES-specific code has been
exercised on the instrument since Ian's May 2024 sessions. Check on
arrival (most are covered by `python -m fpwfsc.tokyo_drift.preflight`,
which is strictly read-only):

- [ ] `vcam1` stream exists and is being written (preflight reads it)
- [ ] `vcam1_dark` naming still right (bench procedure: write a dark
      there before the run; preflight warns if absent)
- [ ] pyMilk `get_data(check=, timeout=)` signature + `multi_recv_data`
      present (preflight inspects the signature)
- [ ] `vampires_control.filters.get_filter_info_dict` importable and
      its dict still has `WAVEAVE` (loop works without it; wavelength
      display only)
- [ ] `FILTER01` keyword format vs mode manifest (preflight compares)
- [ ] Pixel scale: class says 5.9 mas/pix "need to confirm", modes
      trained at 6.0. **The SCExAO wiki warns VAMPIRES is
      non-telecentric — plate scale changes with focus/beamsplitter
      config** — so treat it as per-configuration; a ~1.7% sampling
      mismatch is NOT absorbed by the calibration axes.
- [ ] Frame geometry: class hardcodes 536x536; wiki-era crops were
      128/256/512 windows and the new cameras have MBI crop modes.
      The pipeline doesn't use xsize/ysize, but crop_cx/cy in the
      calibration profile must be re-fit for the actual frame.
- [ ] DM channel: default `dm00disp04` (current FnF deployment); May
      2024 sessions used `dm00disp02`. Confirm with the SCExAO crew
      which channel is allocated to us; it's `[DM] dm channel`.
- [ ] Detector: VAMPIRES is the visible arm — CMOS (wiki specs 0.45 /
      0.25 e- read noise fast/slow, 0.1 e-/ADU), NOT a CRED2 (that's
      Palila). If frames show row/column striping, sc6's
      `support_functions.py` edit has a row+col-median background
      estimator worth porting as a reducer *option* (do not change the
      shared `equalize_image` default — it would alter every pipeline
      including Keck).
- [ ] DM basis diameter: their fnf resamples to 44 actuators (7.48 m);
      our trained basis is 7.79 m (~45.8 act). Centers agree exactly
      (see above); the ~2-actuator edge-taper difference is a modeling
      choice pinned by the NN training, not an error.

## Keck / other-config impact

- `common/bench_hardware.py`: import guards preserve on-Keck behavior
  (imports still succeed there); all class changes are Subaru-only
  classes. Keck aliases untouched.
- fnf untouched (still Keck-shaped; would need a per-instrument branch
  to run at Subaru — upstream's call).
- No changes to qacits / san / satellite.

# 2026-08-24 update — session-log provenance + raw readout capture

Two log-completeness gaps closed ahead of the bench run:

## Session logs are now self-describing

Every logged run copies its calibration profile into the session
directory (`calibration_profile.yaml`) and saves the background/dark
frame the reducer actually subtracted (`background.fits`).
Previously `config.json` recorded the profile only by *path* — for GUI
"run with unsaved parameters" sessions that path is an ephemeral
tempfile in $TMPDIR, so the raw→processed transform (image_rot_deg,
crop center, flips) was unrecoverable from the log alone. The dark
matters for the same reason: its shm buffer is overwritten by the next
dark taken.

## `[IO] save camera frames` (default off)

On hardware, `raw.fits` is the *averaged, dark-subtracted* frame — the
individual readouts are gone, which forecloses per-frame diagnostics
(CMOS striping — see the deferred striping-reducer item — cosmic rays,
dark drift) and any offline re-reduction. With the option on, each
iteration also writes `camera_raw.fits`: the pre-reduction readout cube
in native dtype (`Vampires` now stashes each grab as `last_frames`;
536×536 uint16 × 8 frames ≈ 4.6 MB/iter). Sim and hitchhiker runs have
no camera readout; the option warns and is ignored there.

**Bench-day note:** turn `save camera frames` ON for telescope runs —
the cost is ~130 MB per 28-iteration run against irreplaceable bench
time.

## `[SIMULATION] n repeats` (default 1)

Runs N full loop episodes back to back without rebuilding the expensive
setup (sim construction ~5 s x2, model load ~0.5 s — amortized to
once). Each episode: DM zeroed through the loop's own command path
(the loop images before it commands and never resets the DM itself),
fresh error injection (sim; `set_modal_error` replaces), fresh
integrator + diversity move, own session directory
(`tokyo_drift_<stamp>_rNN` — suffixed because back-to-back sessions
would collide at the 1 s stamp resolution). Honored on hardware too:
re-convergence statistics against the natural NCPA; note the zeroing
discards the previous episode's converged correction from the channel
(it stays in that session's `dm_command.fits` logs). The Strehl
reference is computed once, before the first injection — recomputing it
per episode on the live bench would contaminate the denominator.
Session logs also now carry `episode.json` (episode index, injected
error coefficients, initial diversity move) so `state[0]` is
decomposable offline.

## DM safety trips are now structured episode aborts (not crashes)

Previously a `DMSafetyError` propagated unhandled: the tripping
iteration was never logged (the callback runs after send), no
summary.json was written, remaining repeats died, the GUI thread ended
with only a terminal traceback, and the DM kept the last (near-limit)
sent command. Now the loop catches the refusal: the violating command
is still never sent, but the episode ends with `aborted: "dm_safety"`
in its result/summary, the full forensics are logged (the refused
command as `dm_command_refused.fits` — deliberately NOT
`dm_command.fits`, so offline tools never mistake it for an applied
command — plus `command_sent: false` and the refusal message in that
iteration's metadata), the DM is zeroed, and later episodes continue.
If the first 3 episodes all abort, the run stops (systematic gain /
calibration problem, not unlucky draws). The GUI now surfaces any
loop-thread exception as a dialog. Behavior change for scripts:
`run()` no longer raises `DMSafetyError` — check
`result["loop"]["aborted"]`. `DMSafetyBounds.check` itself still
raises (manual_poke and direct users keep fail-loud semantics), and
the 80% warn threshold is unchanged.
