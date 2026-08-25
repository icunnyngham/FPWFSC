"""Translation DM tests, including a bench-verbatim golden comparison."""
import numpy as np
import pytest

hcipy = pytest.importorskip("hcipy")

from fpwfsc.tokyo_drift.dm import DMSafetyBounds, DMSafetyError, TranslationDM


def _bench_verbatim_surface(dm_act, n_zerns=10, dm_rot_deg=-6.5,
                            scale=1.3e-06, shift_x=1, shift_y=2):
    """The 24-03-27 bench notebook's SubaruZernikeDM, reproduced
    line-for-line as the golden reference for the port."""
    zernike_diameter = 7.79
    dm_spacing_cm = 17
    dm_res = 50
    dm_pupil_extent = dm_res * 0.01 * dm_spacing_cm
    pupil_grid = hcipy.make_pupil_grid(dm_res, dm_pupil_extent)
    if dm_rot_deg is not None:
        pupil_grid = pupil_grid.rotated(dm_rot_deg * np.pi / 180)
    dm_basis = hcipy.make_zernike_basis(n_zerns, zernike_diameter,
                                        pupil_grid, starting_mode=2)
    dm_basis = hcipy.ModeBasis([b / np.max(np.abs(b)) for b in dm_basis])
    dm = hcipy.DeformableMirror(dm_basis)

    dm.actuators = np.asarray(dm_act) * scale
    surf = np.array(dm.surface.reshape((dm_res, dm_res)))
    surf[..., :-shift_x] = surf[..., shift_x:]
    surf[:-shift_y] = surf[shift_y:]
    return surf


def test_port_matches_bench_verbatim_construction():
    acts = np.zeros(10)
    acts[3] = 1.0
    acts[7] = -0.4
    golden = _bench_verbatim_surface(acts)
    # The bench notebooks had no command-aperture crop; disable the
    # (default-on) taper to compare against the verbatim construction.
    ported = TranslationDM(n_modes=10, dm_actuate_scale=1.3e-06,
                           dm_rot_deg=-6.5, shift_x=1, shift_y=2,
                           command_aperture_act=None)
    np.testing.assert_array_equal(ported.act_and_get_surf(acts), golden)


# --- Command-aperture taper --------------------------------------------

def test_default_taper_footprint_matches_deployed_fnf_paste_box():
    """The default 44-actuator crop, shifted by the bench (1, 2), must
    land exactly on the deployed fnf ``make_dm_command`` paste region:
    a 44x44 box centered at actuator (24, 23) on the 50x50 grid."""
    diameter, center = 44, [24, 23]
    x_start = int(center[0] - diameter / 2)
    y_start = int(center[1] - diameter / 2)
    fnf_box = np.zeros((50, 50), dtype=bool)
    fnf_box[y_start:y_start + diameter, x_start:x_start + diameter] = True

    acts = np.zeros(10)
    acts[2] = 1.0  # defocus: nonzero over the whole pupil disk
    dm = TranslationDM(n_modes=10, shift_x=1, shift_y=2)
    surf = dm.act_and_get_surf(acts)
    assert not np.any(surf[~fnf_box])
    assert np.any(surf[fnf_box])


def test_taper_only_trims_outside_the_box():
    """Inside the aperture box the command is untouched."""
    acts = np.random.default_rng(1).normal(size=10)
    tapered = TranslationDM(n_modes=10).act_and_get_surf(acts)
    full = TranslationDM(n_modes=10,
                         command_aperture_act=None).act_and_get_surf(acts)
    lo, hi = (50 - 44) // 2, (50 - 44) // 2 + 44
    np.testing.assert_array_equal(tapered[lo:hi, lo:hi], full[lo:hi, lo:hi])
    outside = np.ones((50, 50), dtype=bool)
    outside[lo:hi, lo:hi] = False
    assert not np.any(tapered[outside])


def test_taper_wider_than_grid_raises():
    with pytest.raises(ValueError):
        TranslationDM(n_modes=10, command_aperture_act=51)


def test_zero_actuation_gives_flat_surface():
    dm = TranslationDM(n_modes=10)
    assert not np.any(dm.act_and_get_surf())


def test_surface_scales_with_actuate_scale():
    acts = np.zeros(10)
    acts[2] = 1.0
    s1 = TranslationDM(n_modes=10, dm_actuate_scale=1e-6).act_and_get_surf(acts)
    s2 = TranslationDM(n_modes=10, dm_actuate_scale=2e-6).act_and_get_surf(acts)
    np.testing.assert_allclose(s2, 2 * s1)


def test_zero_shift_is_a_true_no_op():
    """Guards the `[:-0]` empty-slice pitfall: shift 0 must not touch
    the surface at all (the bench code hardcoded shifts 1 and 2)."""
    acts = np.random.default_rng(0).normal(size=10)
    dm_ref = TranslationDM(n_modes=10, dm_rot_deg=-6.5)
    dm0 = TranslationDM(n_modes=10, dm_rot_deg=-6.5, shift_x=0, shift_y=0)
    np.testing.assert_array_equal(dm0.act_and_get_surf(acts),
                                  dm_ref.act_and_get_surf(acts))


def test_flips_mirror_surface():
    acts = np.zeros(10)
    acts[0] = 1.0  # tilt: asymmetric surface
    base = TranslationDM(n_modes=10).act_and_get_surf(acts)
    flipped_v = TranslationDM(n_modes=10, flip_vertical=True).act_and_get_surf(acts)
    flipped_h = TranslationDM(n_modes=10, flip_horizontal=True).act_and_get_surf(acts)
    np.testing.assert_array_equal(flipped_v, np.flipud(base))
    np.testing.assert_array_equal(flipped_h, np.fliplr(base))


def test_command_microns_units_and_dtype():
    acts = np.zeros(10)
    acts[2] = 1.0
    dm = TranslationDM(n_modes=10, dm_actuate_scale=1e-6)
    surf_m = dm.act_and_get_surf(acts)
    cmd = dm.command_microns(acts)
    assert cmd.dtype == np.float32
    np.testing.assert_allclose(cmd, surf_m * 1e6, rtol=1e-6)


def test_wrong_coefficient_count_raises():
    with pytest.raises(ValueError):
        TranslationDM(n_modes=10).act_and_get_surf(np.zeros(11))


# --- Safety bounds -----------------------------------------------------

def test_safety_passes_small_command(capsys):
    bounds = DMSafetyBounds(max_ptv_um=2.0, max_stroke_um=1.0)
    cmd = np.full((50, 50), 0.1)
    assert bounds.check(cmd) is cmd
    assert "WARNING" not in capsys.readouterr().out


def test_safety_warns_near_limit(capsys):
    bounds = DMSafetyBounds(max_ptv_um=2.0, max_stroke_um=1.0)
    cmd = np.zeros((50, 50))
    cmd[25, 25] = 0.9  # 90% of stroke limit
    bounds.check(cmd)
    assert "WARNING" in capsys.readouterr().out


def test_safety_refuses_over_stroke():
    bounds = DMSafetyBounds(max_ptv_um=5.0, max_stroke_um=1.0)
    cmd = np.zeros((50, 50))
    cmd[25, 25] = 1.5
    with pytest.raises(DMSafetyError):
        bounds.check(cmd)


def test_safety_refuses_over_ptv():
    bounds = DMSafetyBounds(max_ptv_um=1.0, max_stroke_um=1.0)
    cmd = np.zeros((50, 50))
    cmd[0, 0] = -0.8
    cmd[49, 49] = 0.8  # each within stroke, PTV 1.6 over limit
    with pytest.raises(DMSafetyError):
        bounds.check(cmd)
