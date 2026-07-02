"""Translation DM: mode coefficients -> raw 50x50 DM command.

The HCIPy-DM-as-Zernike-translator trick, ported from the SCExAO bench
sessions (where it ran closed-loop as ``SubaruZernikeDM``): an HCIPy
``DeformableMirror`` is built on a pupil grid whose resolution *is* the
actuator-grid resolution, so ``dm.surface`` reads out directly as the
50x50 command array — no resampling step. The Zernike basis is
constructed with the same call and max-abs normalization as the
training simulator, so an NN-predicted coefficient vector is
dimensionally identical on sim and bench; only ``dm_actuate_scale``
mediates the meters-per-coefficient calibration (bench-validated values
were 1.3e-6 to 1.4e-6 across 2024 sessions).

Misalignment handling (all sourced from the calibration profile):

- ``dm_rot_deg`` is baked into the *basis* at construction via
  ``pupil_grid.rotated()`` — surfaces come out already rotated, with no
  per-command rotation step.
- ``flip_vertical`` / ``flip_horizontal`` mirror the output surface.
- ``shift_x`` / ``shift_y`` integer-pixel shifts reproduce the bench
  workaround for the DM pupil sitting off-center on the 50x50 actuator
  grid. **Default 0 (off)** — they are a bench-hardware artifact; the
  simulated bench deliberately has no shift mechanism, and sim-mode
  calibration profiles must not set them.
"""
import numpy as np


class TranslationDM:
    """Zernike-coefficient to actuator-command translator.

    Parameters
    ----------
    n_modes
        Number of Zernike modes (Noll indexing from ``starting_mode``).
    dm_actuate_scale
        Meters of DM surface per unit coefficient (calibration
        parameter; bench-validated ~1.4e-6).
    dm_rot_deg
        DM rotation relative to the model, degrees, baked into the
        basis. ``None`` (or 0) skips the grid rotation entirely.
    flip_vertical, flip_horizontal
        Mirror the output surface (``flipud`` / ``fliplr``).
    shift_x, shift_y
        Integer-pixel command shifts (see module docstring). Default 0.
    zernike_diameter, dm_spacing_cm, dm_res, starting_mode
        Geometry constants; defaults are the SCExAO/VAMPIRES values the
        bench sessions used (7.79 m basis on a 50x50 grid at 17 cm
        projected pitch -> 8.5 m grid extent).
    """

    def __init__(
        self,
        n_modes=10,
        dm_actuate_scale=1.4e-06,
        dm_rot_deg=None,
        flip_vertical=False,
        flip_horizontal=False,
        shift_x=0,
        shift_y=0,
        zernike_diameter=7.79,
        dm_spacing_cm=17,
        dm_res=50,
        starting_mode=2,
    ):
        import hcipy

        self.n_modes = int(n_modes)
        self.dm_actuate_scale = float(dm_actuate_scale)
        self.dm_rot_deg = dm_rot_deg
        self.flip_v = bool(flip_vertical)
        self.flip_h = bool(flip_horizontal)
        self.shift_x = int(shift_x)
        self.shift_y = int(shift_y)
        self.dm_shape = (int(dm_res), int(dm_res))

        dm_pupil_extent = dm_res * 0.01 * dm_spacing_cm
        pupil_grid = hcipy.make_pupil_grid(dm_res, dm_pupil_extent)
        if self.dm_rot_deg is not None:
            pupil_grid = pupil_grid.rotated(self.dm_rot_deg * np.pi / 180)
        self.pupil_grid = pupil_grid
        basis = hcipy.make_zernike_basis(
            self.n_modes, zernike_diameter, pupil_grid,
            starting_mode=starting_mode)
        basis = hcipy.ModeBasis([b / np.max(np.abs(b)) for b in basis])
        self.dm = hcipy.DeformableMirror(basis)

    def act_and_get_surf(self, dm_act=None):
        """Set mode coefficients, return the 50x50 surface in meters."""
        if dm_act is None:
            dm_act = np.zeros((self.n_modes,))
        dm_act = np.asarray(dm_act, dtype=float)
        if dm_act.shape != (self.n_modes,):
            raise ValueError(
                f"expected ({self.n_modes},) coefficients, got {dm_act.shape}")

        self.dm.actuators = dm_act * self.dm_actuate_scale
        surf = np.array(self.dm.surface.reshape(self.dm_shape))

        if self.flip_v:
            surf = np.flipud(surf)
        if self.flip_h:
            surf = np.fliplr(surf)

        # Bench-verbatim integer shifts (guarded so 0 means "off";
        # the bench code hardcoded 1 and 2 and used `is not None`).
        if self.shift_x:
            surf[..., :-self.shift_x] = surf[..., self.shift_x:]
        if self.shift_y:
            surf[:-self.shift_y] = surf[self.shift_y:]

        return surf

    def command_microns(self, dm_act=None):
        """The 50x50 command in microns as float32 — exactly what
        ``set_dm_data`` ships to the shared-memory stream."""
        return (self.act_and_get_surf(dm_act) * 1e6).astype(np.float32)
