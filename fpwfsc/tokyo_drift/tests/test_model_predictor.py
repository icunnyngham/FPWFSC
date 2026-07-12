"""Tests for the trained-NN predictor path.

Three tiers, degrading gracefully by what's installed/available:

- **Unit** (needs torch only): a tiny throwaway ``FFModelTorch`` is saved
  and loaded through the real plumbing, exercising checkpoint resolution,
  the predict contract (shape/dtype), and — critically — the actuation
  sign flip that bridges the loop's command-delta convention to the
  model's training convention.
- **Sign validation** (needs the real checkpoint + telescope_sim): one
  eval step reproduced on the training-matched ideal sim, where the model
  is near-perfect. Confirms both signs: the prediction tracks the true
  residual, and the correct actuation sign beats the flipped one.
- **Convergence** (needs the real checkpoint + telescope_sim): the loop
  driven by the model on the ``easy`` bench preset improves the Strehl.
  The bench sim deliberately differs from the training sim (influence-
  function DM, mangling, noise — the inverse-crime discipline), so this
  is not oracle-perfect; improvement, not perfection, is the bar.
"""
from pathlib import Path

import numpy as np
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]

# Registered NN modes, with the per-mode knobs the checkpoint-dependent
# tests need. ``err_rms`` is a training-appropriate injected error (the
# model is only in-distribution up to ~its training sigma: 0.05 for F760,
# 0.03 for F750 — 0.15 puts the 35-mode model far OOD); ``min_cos`` is the
# measured ideal-sim floor (10 modes fit near-perfectly, 35 a touch less).
MODES = [
    {"name": "vampires_f760_10zern", "n_modes": 10, "err_rms": 0.15,
     "min_cos": 0.99, "coro": False},
    # The 35-mode model fits the ideal sim a touch less tightly (measured
    # ~0.94); the flipped sign there collapses to ~-0.09, so the sign check
    # is still decisive.
    {"name": "vampires_f750_35zern", "n_modes": 35, "err_rms": 0.05,
     "min_cos": 0.90, "coro": False},
    # VVC coronagraph model (charge 4). Same FFModel family; fits the ideal
    # sim ~0.89 (flipped sign ~-0.07, still decisive) and renders a 120px
    # focal plane natively (model input_hw=120). Excluded from the
    # Strehl-gated convergence test (coro Strehl is only a leakage proxy);
    # it has its own modal-residual convergence test below.
    {"name": "vampires_vvc_f750_35zern_crop", "n_modes": 35, "err_rms": 0.05,
     "min_cos": 0.85, "coro": True},
]

NONCORO_MODES = [m for m in MODES if not m["coro"]]


def _require_checkpoint(name):
    from fpwfsc.tokyo_drift.mode_registry import checkpoint_path
    try:
        checkpoint_path(name)
    except (ValueError, FileNotFoundError, KeyError):
        pytest.skip(f"trained checkpoint for {name} not present (gitignored)")


def _tiny_checkpoint(tmp_path):
    """Save a small but architecturally-real FFModelTorch checkpoint."""
    import torch

    from fpwfsc.tokyo_drift.model_torch import FFModelTorch

    meta = dict(n_modes=10, conv_channels=4, conv_kernel=5, n_conv=4,
                dense_size=8, n_dense=2, psf_channels=2, run_id="unit-test")
    model = FFModelTorch(**{k: meta[k] for k in
                            ("n_modes", "conv_channels", "conv_kernel",
                             "n_conv", "dense_size", "n_dense",
                             "psf_channels")})
    path = tmp_path / "tiny.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta}, str(path))
    return str(path)


# --- Unit: plumbing + sign flip -----------------------------------------

def test_predict_contract_and_shape(tmp_path):
    pytest.importorskip("torch")
    from fpwfsc.tokyo_drift.model_predictor import TorchPredictor

    pred = TorchPredictor(_tiny_checkpoint(tmp_path))
    assert pred.n_modes == 10
    assert pred.run_id == "unit-test"

    frames = np.random.default_rng(0).random((2, 128, 128)).astype(np.float32)
    out = pred.predict(frames, np.zeros(10))
    assert out.shape == (10,)
    assert out.dtype == np.float64
    assert pred.last_latency_s is not None and pred.last_latency_s >= 0


def test_actuation_sign_is_negated(tmp_path):
    """The adapter feeds the model ``move = -delta_actuation`` (loop's
    command-delta convention -> model's training convention)."""
    pytest.importorskip("torch")
    from fpwfsc.tokyo_drift.model_predictor import TorchPredictor

    pred = TorchPredictor(_tiny_checkpoint(tmp_path))
    rng = np.random.default_rng(1)
    frames = rng.random((2, 128, 128)).astype(np.float32)
    actuation = rng.normal(0, 0.1, 10)

    via_adapter = pred.predict(frames, actuation)
    # Reference: call the model directly with the negated actuation.
    via_model = pred.model.predict_residual(frames[0], frames[1], -actuation,
                                            device="cpu")
    np.testing.assert_allclose(via_adapter, via_model, rtol=1e-6, atol=1e-8)

    # And the flipped input genuinely differs (sign wiring is not a no-op).
    flipped = pred.model.predict_residual(frames[0], frames[1], actuation,
                                          device="cpu")
    assert not np.allclose(via_adapter, flipped, atol=1e-6)


def test_missing_checkpoint_errors_clearly():
    from fpwfsc.tokyo_drift.mode_registry import checkpoint_path
    # An unknown mode has no manifest -> null checkpoint -> clear ValueError.
    with pytest.raises((ValueError, KeyError)):
        checkpoint_path("no_such_mode_xyz")


# --- Sign validation on the training-matched ideal sim ------------------

@pytest.mark.parametrize("mode", MODES, ids=lambda m: m["name"])
def test_prediction_tracks_true_residual_on_ideal_sim(mode):
    """On the training-matched sim (no mangling) the model tracks the true
    residual with the correct actuation sign, beating the flipped one."""
    pytest.importorskip("torch")
    pytest.importorskip("telescope_sim")
    _require_checkpoint(mode["name"])
    from fpwfsc.tokyo_drift.model_predictor import TorchPredictor
    from fpwfsc.tokyo_drift.sim import IdealSim

    n = mode["n_modes"]
    ideal = IdealSim.from_mode(mode["name"])
    pred = TorchPredictor.from_mode(mode["name"])
    rng = np.random.default_rng(0)

    def psf(z):
        return np.asarray(ideal.psf(actuations={"zernike_dm": np.asarray(z)}),
                          dtype=np.float32)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    right, wrong = [], []
    for _ in range(6):
        err = rng.normal(0.0, 0.05, n)
        move = rng.normal(0.0, 0.05, n)           # large enough that sign matters
        target = err - move                        # current residual (state_after)
        p0, p1 = psf(err), psf(err - move)

        frames = np.stack([p0, p1])
        # Loop reports command_after - command_before = (err-move) - err = -move.
        delta_actuation = -move
        r = pred.predict(frames, delta_actuation)  # adapter negates -> move=+move
        right.append(cos(r, target))

        r_flip = pred.model.predict_residual(p0, p1, -move, device="cpu")
        wrong.append(cos(r_flip, target))

    assert np.mean(right) > mode["min_cos"]        # tracks the true residual
    assert np.mean(right) > np.mean(wrong)         # correct sign beats flipped


# --- End-to-end convergence on the bench sim ----------------------------

@pytest.fixture(scope="module")
def fitted_profile_for(tmp_path_factory):
    """Factory: a fitted easy-preset profile per mode, calibrated once and
    cached (calibration is the expensive step)."""
    pytest.importorskip("telescope_sim")
    yaml = pytest.importorskip("yaml")
    from fpwfsc.tokyo_drift.calibration.harness import calibrate_bench_sim
    cache = {}

    def get(mode_name):
        if mode_name not in cache:
            profile, _ = calibrate_bench_sim(mode_name, preset="easy", seed=27)
            path = tmp_path_factory.mktemp("cal") / f"{mode_name}.yaml"
            path.write_text(yaml.safe_dump(profile))
            cache[mode_name] = str(path)
        return cache[mode_name]

    return get


@pytest.mark.parametrize("mode", NONCORO_MODES, ids=lambda m: m["name"])
def test_model_loop_improves_strehl(mode, fitted_profile_for):
    """The model-driven loop converges on the easy bench preset.

    Not oracle-perfect by design (the bench sim differs from the training
    sim). ``err_rms`` is kept near the model's training scale — a much
    larger injected error is out-of-distribution and can trip DM safety
    (notably for the 35-mode model). Measured levels are in
    tokyo_drift_plan.md.
    """
    pytest.importorskip("torch")
    _require_checkpoint(mode["name"])
    from configobj import ConfigObj

    from fpwfsc.tokyo_drift.run import run

    cfg = ConfigObj(str(PIPELINE_DIR / "tokyo_drift_config_sim.ini"))
    cfg["MODE"]["mode name"] = mode["name"]
    cfg["LOOP_SETTINGS"]["predictor"] = "model"
    cfg["LOOP_SETTINGS"]["N iter"] = 12
    cfg["SIMULATION"]["initial error rms"] = str(mode["err_rms"])
    cfg["MODE"]["calibration profile"] = fitted_profile_for(mode["name"])
    result = run("Sim", "Sim", config=cfg,
                 configspec=str(PIPELINE_DIR / "tokyo_drift_config.spec"))

    strehls = result["loop"]["strehls"]
    valid = strehls[np.isfinite(strehls)]
    assert valid[0] < 0.6                        # starts aberrated
    # Measured (seed 27, easy): 10z reaches ~1.03, 35z ~0.96-0.99. Floor
    # pinned well below; bench-vs-training mismatch caps the ceiling.
    assert valid.max() > 0.85


def test_coro_model_loop_converges_modal_residual(fitted_profile_for):
    """The VVC model closes the loop on the actuator-grid bench.

    End-to-end regression for the 2026-07 coro-divergence work: this
    needs the gain-aware calibration (dm_scale ~1.7, not ~1.0) AND SNR
    mitigation — a coronagraph's faint speckle field diverges the model
    on single noisy frames at the training-era flux, so the loop runs
    with frame averaging (the sim-flux knob works equally; see
    MODEL_INTEGRATION_NOTES). Gated on the true modal residual
    ``||err + state||``; coro Strehl is only a leakage proxy.
    """
    pytest.importorskip("torch")
    mode = next(m for m in MODES if m["coro"])
    _require_checkpoint(mode["name"])
    from configobj import ConfigObj

    from fpwfsc.tokyo_drift.run import run

    cfg = ConfigObj(str(PIPELINE_DIR / "tokyo_drift_config_sim.ini"))
    cfg["MODE"]["mode name"] = mode["name"]
    cfg["LOOP_SETTINGS"]["predictor"] = "model"
    cfg["LOOP_SETTINGS"]["N iter"] = 10
    cfg["LOOP_SETTINGS"]["strehl early stop"] = "None"
    cfg["SIMULATION"]["initial error rms"] = "0.02"
    cfg["SNR"]["frames to average"] = "8"
    cfg["MODE"]["calibration profile"] = fitted_profile_for(mode["name"])
    result = run("Sim", "Sim", config=cfg,
                 configspec=str(PIPELINE_DIR / "tokyo_drift_config.spec"))

    err = result["injected_error_coeffs"]
    residuals = [float(np.linalg.norm(err + s))
                 for s in result["loop"]["states"]]
    assert np.linalg.norm(err) > 0.08            # starts aberrated
    assert min(residuals) < 0.03                 # converges (measured ~0.014)
    assert residuals[-1] < 0.04                  # and stays converged
