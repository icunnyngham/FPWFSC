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
MODE = "vampires_f760_10zern"


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

def _checkpoint_available():
    from fpwfsc.tokyo_drift.mode_registry import checkpoint_path
    try:
        checkpoint_path(MODE)
        return True
    except (ValueError, FileNotFoundError, KeyError):
        return False


needs_checkpoint = pytest.mark.skipif(
    not _checkpoint_available(),
    reason=f"trained checkpoint for {MODE} not present (gitignored)")


@needs_checkpoint
def test_prediction_tracks_true_residual_on_ideal_sim():
    """On the training-matched sim (no mangling) the model is near-perfect,
    and the correct actuation sign beats the flipped one."""
    pytest.importorskip("torch")
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.model_predictor import TorchPredictor
    from fpwfsc.tokyo_drift.sim import IdealSim

    ideal = IdealSim.from_mode(MODE)
    pred = TorchPredictor.from_mode(MODE)
    rng = np.random.default_rng(0)

    def psf(z):
        return np.asarray(ideal.psf(actuations={"zernike_dm": np.asarray(z)}),
                          dtype=np.float32)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    right, wrong = [], []
    for _ in range(6):
        err = rng.normal(0.0, 0.05, 10)
        move = rng.normal(0.0, 0.05, 10)          # large enough that sign matters
        target = err - move                        # current residual (state_after)
        p0, p1 = psf(err), psf(err - move)

        frames = np.stack([p0, p1])
        # Loop reports command_after - command_before = (err-move) - err = -move.
        delta_actuation = -move
        r = pred.predict(frames, delta_actuation)  # adapter negates -> move=+move
        right.append((cos(r, target), np.linalg.norm(r - target)))

        r_flip = pred.model.predict_residual(p0, p1, -move, device="cpu")
        wrong.append(cos(r_flip, target))

    right = np.array(right)
    assert right[:, 0].mean() > 0.99          # near-perfect on training-matched sim
    assert right[:, 1].mean() < 0.02          # <2% residual-vector error
    assert right[:, 0].mean() > np.mean(wrong)  # correct sign beats flipped


# --- End-to-end convergence on the bench sim ----------------------------

@pytest.fixture(scope="module")
def fitted_profile(tmp_path_factory):
    pytest.importorskip("telescope_sim")
    yaml = pytest.importorskip("yaml")
    from fpwfsc.tokyo_drift.calibration.harness import calibrate_bench_sim
    profile, _ = calibrate_bench_sim(MODE, preset="easy", seed=27)
    path = tmp_path_factory.mktemp("calibrations") / "fitted_easy_27.yaml"
    path.write_text(yaml.safe_dump(profile))
    return str(path)


@needs_checkpoint
def test_model_loop_improves_strehl(fitted_profile):
    """The model-driven loop improves the Strehl on the easy bench preset.

    Not oracle-perfect by design (the bench sim differs from the training
    sim); the measured achievement is recorded in tokyo_drift_plan.md.
    """
    pytest.importorskip("torch")
    from configobj import ConfigObj

    from fpwfsc.tokyo_drift.run import run

    cfg = ConfigObj(str(PIPELINE_DIR / "tokyo_drift_config_sim.ini"))
    cfg["LOOP_SETTINGS"]["predictor"] = "model"
    cfg["LOOP_SETTINGS"]["N iter"] = 12
    cfg["MODE"]["calibration profile"] = fitted_profile
    result = run("Sim", "Sim", config=cfg,
                 configspec=str(PIPELINE_DIR / "tokyo_drift_config.spec"))

    strehls = result["loop"]["strehls"]
    valid = strehls[np.isfinite(strehls)]
    assert valid[0] < 0.5                       # starts aberrated
    # Measured (seed 27, easy preset): the loop walks the 0.15-rms error
    # (~3x the model's 0.05 training scale) down into distribution over the
    # first ~5 steps, then converges hard to Strehl ~1.03. Floor pinned well
    # below that; the bench-vs-training mismatch is what caps the ceiling.
    assert valid.max() > 0.85
