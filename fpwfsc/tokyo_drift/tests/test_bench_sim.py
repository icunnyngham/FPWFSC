"""Bench-sim tests: presets, config derivation, mangling, integration."""
import numpy as np
import pytest

from fpwfsc.tokyo_drift.sim.bench_sim import (
    derive_bench_config,
    load_preset,
    mangle_frame,
    sample_truth,
)

SHIPPED_PRESETS = ["easy", "realistic_vampires", "stress_test", "vampires_2024_measured"]
TRUTH_KEYS = {"image_rot_deg", "dm_rot_deg", "crop_dx", "crop_dy",
              "dm_scale", "dm_flip_x", "dm_flip_y"}


# --- Presets -----------------------------------------------------------

@pytest.mark.parametrize("name", SHIPPED_PRESETS)
def test_shipped_presets_load_and_sample(name):
    pytest.importorskip("yaml")
    preset = load_preset(name)
    truth = sample_truth(preset, np.random.default_rng(0))
    assert set(truth) == TRUTH_KEYS


def test_unknown_preset_raises():
    pytest.importorskip("yaml")
    with pytest.raises(KeyError):
        load_preset("no_such_preset")


def test_vampires_2024_preset_is_deterministic():
    pytest.importorskip("yaml")
    preset = load_preset("vampires_2024_measured")
    t1 = sample_truth(preset, np.random.default_rng(1))
    t2 = sample_truth(preset, np.random.default_rng(2))
    assert t1 == t2
    assert t1["image_rot_deg"] == 235.4
    assert t1["dm_rot_deg"] == -6.25
    assert t1["dm_scale"] == 1.3
    assert (t1["crop_dx"], t1["crop_dy"]) == (7, 7)
    assert not t1["dm_flip_x"] and not t1["dm_flip_y"]


# --- Config derivation --------------------------------------------------

@pytest.fixture()
def base_config():
    yaml = pytest.importorskip("yaml")
    from fpwfsc.tokyo_drift.mode_registry import ts2_config_path
    with open(ts2_config_path("vampires_f760_10zern")) as f:
        return yaml.safe_load(f)


@pytest.fixture()
def truth():
    return {"image_rot_deg": 90.0, "dm_rot_deg": -6.25, "crop_dx": 3,
            "crop_dy": -2, "dm_scale": 1.3, "dm_flip_x": False,
            "dm_flip_y": True}


def test_derivation_preserves_training_axes(base_config, truth):
    derived = derive_bench_config(base_config, truth, render_res=512)
    base_fp = base_config["focal_planes"]["filter1"]
    fp = derived["focal_planes"]["filter1"]
    # Pupil / filter / bandwidth byte-identical
    assert derived["aperture"] == base_config["aperture"]
    assert fp["central_lam"] == base_fp["central_lam"]
    assert fp["fractional_bandwidth"] == base_fp["fractional_bandwidth"]
    assert fp["num_samples"] == base_fp["num_samples"]
    # Same angular pixel scale at the higher resolution
    assert fp["focal_res"] == 512
    assert fp["focal_extent"] / fp["focal_res"] == pytest.approx(
        base_fp["focal_extent"] / base_fp["focal_res"])


def test_derivation_swaps_corrector_and_strips_post(base_config, truth):
    derived = derive_bench_config(base_config, truth)
    assert derived["corrector_chain"] == ["bench_dm"]
    dm = derived["correctors"]["bench_dm"]
    assert dm["type"] == "actuator_grid"
    assert dm["num_actuators"] == 50
    assert dm["rotation_deg"] == truth["dm_rot_deg"]
    assert dm["flip_y"] is True
    assert dm["actuate_scale"] == pytest.approx(1.0e-6 * 1.3)
    # Pupil grid covers the full actuator lattice
    assert derived["pupil"]["extent"] >= 8.5
    # Raw intensity out: no post-processing anywhere
    for output in derived["outputs"].values():
        assert "post_processing" not in output


def test_derivation_does_not_mutate_base(base_config, truth):
    import copy
    snapshot = copy.deepcopy(base_config)
    derive_bench_config(base_config, truth)
    assert base_config == snapshot


# --- Mangling layer -----------------------------------------------------

def _blob_field(size=512, row=200, col=300):
    """A small Gaussian blob off-center in an otherwise dark field."""
    yy, xx = np.mgrid[:size, :size]
    return np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * 3.0 ** 2))


def _peak(frame):
    return np.unravel_index(frame.argmax(), frame.shape)


def test_mangle_rotates_content():
    field = _blob_field(row=256, col=356)  # +100 px along +x from center
    frame = mangle_frame(field, 90.0, 0, 0, 256)
    # scipy rotates CCW in array (row, col) convention with angle > 0:
    # content at (+100, 0) relative to center moves along -row.
    row, col = _peak(frame)
    assert abs(col - 128) <= 1
    assert abs(row - (128 - 100)) <= 1


def test_mangle_crop_offset_shifts_content():
    field = _blob_field(row=256, col=256)  # centered blob
    frame = mangle_frame(field, 0.0, 10, -20, 256)
    row, col = _peak(frame)
    # Crop center moved to (+10 x, -20 y): content appears displaced
    # the opposite way within the frame.
    assert abs(col - (128 - 10)) <= 1
    assert abs(row - (128 + 20)) <= 1


def test_mangle_noiseless_is_deterministic():
    field = _blob_field()
    a = mangle_frame(field, 33.3, 5, -7, 256)
    b = mangle_frame(field, 33.3, 5, -7, 256)
    np.testing.assert_array_equal(a, b)


def test_mangle_noise_applied_after_scaling():
    field = _blob_field(row=256, col=256)
    total = 1e5
    clean = mangle_frame(field, 0.0, 0, 0, 256, total_photons=total)
    noisy = mangle_frame(field, 0.0, 0, 0, 256,
                         rng=np.random.default_rng(0),
                         total_photons=total, read_noise=5.0)
    # Flux scaling: the noiseless frame carries (nearly) all the photons
    assert clean.sum() == pytest.approx(total, rel=1e-3)
    # Noise present but unbiased: totals agree statistically
    assert not np.array_equal(clean, noisy)
    assert noisy.sum() == pytest.approx(clean.sum(), rel=0.05)


def test_mangle_rejects_crop_outside_field():
    field = _blob_field(size=300)
    with pytest.raises(ValueError):
        mangle_frame(field, 0.0, 100, 0, 256)


# --- Integration with telescope-sim -------------------------------------

@pytest.fixture(scope="module")
def bench():
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.sim import BenchSim
    return BenchSim.from_mode("vampires_f760_10zern",
                              preset="vampires_2024_measured", seed=42)


def test_bench_truth_matches_fixed_preset(bench):
    assert bench.truth["image_rot_deg"] == 235.4
    assert bench.truth["dm_rot_deg"] == -6.25
    assert bench.truth["dm_scale"] == 1.3


def test_bench_take_image_shape(bench):
    frame = bench.take_image()
    assert frame.shape == (256, 256)
    assert np.isfinite(frame).all()


def test_bench_dm_poke_changes_image(bench):
    flat = bench.take_image_noiseless()
    cmd = np.zeros((50, 50))
    cmd[20:30, 20:30] = 0.3  # a patch poke, microns
    bench.set_dm_data(cmd)
    poked = bench.take_image_noiseless()
    bench.set_dm_data(np.zeros((50, 50)))
    assert not np.allclose(flat, poked)


def test_bench_rejects_wrong_command_shape(bench):
    with pytest.raises(ValueError):
        bench.set_dm_data(np.zeros((44, 44)))


def test_bench_average_reduces_noise(bench):
    one = bench.take_image(average=1)
    many = bench.take_image(average=16)
    ref = bench.take_image_noiseless()
    assert np.abs(many - ref).mean() < np.abs(one - ref).mean()
