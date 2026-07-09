"""Closed-loop tests: integrator math, predictors, and the M5
integration checkpoint (oracle converges, random walk does not,
safety bounds trip)."""
import threading
from pathlib import Path

import numpy as np
import pytest
from configobj import ConfigObj

from fpwfsc.tokyo_drift.loop import LeakyIntegrator, manual_poke
from fpwfsc.tokyo_drift.predictors import CheatingOracle, RandomWalkPredictor

PIPELINE_DIR = Path(__file__).resolve().parents[1]
SPEC = str(PIPELINE_DIR / "tokyo_drift_config.spec")
SIM_INI = str(PIPELINE_DIR / "tokyo_drift_config_sim.ini")


def _sim_config(**overrides):
    """The shipped sim ini with LOOP_SETTINGS/SIMULATION overrides."""
    cfg = ConfigObj(SIM_INI)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        cfg[section][key] = value
    return cfg


# --- Unit: integrator + predictors --------------------------------------

def test_leaky_integrator_math():
    integ = LeakyIntegrator(3, gain=0.5, leak=0.9)
    s1 = integ.update(np.array([1.0, 0.0, -2.0]))
    np.testing.assert_allclose(s1, [-0.5, 0.0, 1.0])
    s2 = integ.update(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(s2, [-0.45, -0.5, 0.9])


def test_random_walk_predictor_shape_and_randomness():
    pred = RandomWalkPredictor(10, step_sigma=0.1, seed=0)
    a = pred.predict(None, None)
    b = pred.predict(None, None)
    assert a.shape == (10,)
    assert not np.array_equal(a, b)


def test_cheating_oracle_points_at_residual():
    residual = np.array([0.5, -0.2, 0.1])
    oracle = CheatingOracle(lambda: residual, fuzz_sigma=0.05, seed=0)
    pred = oracle.predict(None, None)
    # Fuzzed but in the known right direction, close to the truth
    assert np.dot(pred, residual) > 0
    np.testing.assert_allclose(pred, residual, rtol=0.25)


def test_manual_poke_sends_and_returns_frame():
    from fpwfsc.tokyo_drift.dm import TranslationDM
    pytest.importorskip("hcipy")
    sent = []
    frame = np.arange(16.0).reshape(4, 4)
    result = manual_poke(
        np.zeros(10),
        take_image=lambda average=1: frame,
        send_command=sent.append,
        translator=TranslationDM(n_modes=10),
    )
    assert len(sent) == 1 and sent[0].shape == (50, 50)
    np.testing.assert_array_equal(result, frame)


# --- Integration: the M5 checkpoint --------------------------------------
#
# The loop requires a calibration profile, and since the injected loop
# error is EXTERNAL (NCPA-like), cancelling it needs the DM's effective
# command-to-wavefront gain — which only a real fit measures (the
# injected truth scale alone misses the influence-function gain). So
# these tests calibrate once and share the fitted profile.

@pytest.fixture(scope="module")
def fitted_profile(tmp_path_factory):
    pytest.importorskip("telescope_sim")
    yaml = pytest.importorskip("yaml")
    from fpwfsc.tokyo_drift.calibration.harness import calibrate_bench_sim
    profile, _report = calibrate_bench_sim("vampires_f760_10zern",
                                           preset="easy", seed=27)
    path = tmp_path_factory.mktemp("calibrations") / "fitted_easy_27.yaml"
    path.write_text(yaml.safe_dump(profile))
    return str(path)


@pytest.fixture(scope="module")
def oracle_result(fitted_profile):
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cfg = _sim_config(**{"LOOP_SETTINGS.N iter": 15,
                         "LOOP_SETTINGS.predictor": "oracle",
                         "LOOP_SETTINGS.strehl early stop": 0.95,
                         "MODE.calibration profile": fitted_profile})
    return run("Sim", "Sim", config=cfg, configspec=SPEC)


def test_oracle_loop_converges(oracle_result):
    """THE integration checkpoint: an oracle-driven loop must converge."""
    loop = oracle_result["loop"]
    strehls = loop["strehls"]
    valid = strehls[np.isfinite(strehls)]
    assert valid[0] < 0.3          # starts aberrated
    assert valid[-1] >= 0.95       # converges (proxy ~1 at reference)

    # The converged state cancels the injected error
    error = oracle_result["injected_error_coeffs"]
    final = loop["final_state"]
    cos = np.dot(final, -error) / (
        np.linalg.norm(final) * np.linalg.norm(error))
    assert cos > 0.99


def test_oracle_loop_early_stops(oracle_result):
    # strehl early stop 0.95 fires well before the 15-iteration cap
    assert oracle_result["loop"]["iterations"] < 15


def test_random_walk_loop_does_not_converge_and_logs(fitted_profile,
                                                     tmp_path):
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cfg = _sim_config(**{"LOOP_SETTINGS.N iter": 5,
                         "LOOP_SETTINGS.predictor": "random_walk",
                         "MODE.calibration profile": fitted_profile,
                         "IO.save_log": True,
                         "IO.log_path": str(tmp_path)})
    result = run("Sim", "Sim", config=cfg, configspec=SPEC)
    strehls = result["loop"]["strehls"]
    assert np.nanmax(strehls) < 0.5

    # Session log written: one timestamped dir with config, per-iter
    # records, and a summary.
    sessions = list(tmp_path.glob("tokyo_drift_*"))
    assert len(sessions) == 1
    session = sessions[0]
    assert (session / "config.json").is_file()
    assert (session / "summary.json").is_file()
    iter_dirs = sorted(p.name for p in session.glob("iter_*"))
    assert iter_dirs == [f"iter_{i:03d}" for i in range(5)]
    assert (session / "iter_000" / "dm_command.fits").is_file()


def test_safety_bounds_trip_on_oversized_command(fitted_profile):
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.dm import DMSafetyError
    from fpwfsc.tokyo_drift.run import run
    cfg = _sim_config(**{"LOOP_SETTINGS.N iter": 3,
                         "LOOP_SETTINGS.predictor": "oracle",
                         "DM.max actuator stroke (um)": 0.001,
                         "MODE.calibration profile": fitted_profile})
    with pytest.raises(DMSafetyError):
        run("Sim", "Sim", config=cfg, configspec=SPEC)


def test_run_requires_calibration_profile():
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cfg = _sim_config()  # profile stays None
    with pytest.raises(ValueError, match="calibration profile"):
        run("Sim", "Sim", config=cfg, configspec=SPEC)


def test_stop_event_interrupts_loop(fitted_profile):
    pytest.importorskip("telescope_sim")
    # Event set after validation would skip the loop entirely; instead
    # verify the mid-loop check: one iteration completes when the event
    # is set by the first plotter callback.
    class StopOnFirstUpdate:
        def __init__(self, event):
            self.event = event

        def update(self, payload):
            self.event.set()

    event = threading.Event()
    cfg = _sim_config(**{"LOOP_SETTINGS.N iter": 10,
                         "LOOP_SETTINGS.predictor": "oracle",
                         "MODE.calibration profile": fitted_profile})
    from fpwfsc.tokyo_drift import run as run_mod
    result = run_mod.run("Sim", "Sim", config=cfg, configspec=SPEC,
                         my_event=event, plotter=StopOnFirstUpdate(event))
    assert result["loop"]["iterations"] == 1
