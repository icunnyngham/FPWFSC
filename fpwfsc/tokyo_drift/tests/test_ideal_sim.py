"""Ideal-sim backend tests (require telescope-sim)."""
import numpy as np
import pytest

pytest.importorskip("telescope_sim")

from fpwfsc.tokyo_drift.sim import IdealSim


@pytest.fixture(scope="module")
def ideal():
    # Built once per module: TS2 pipeline construction (supersampled
    # aperture + Zernike bind on the 256^2 grid) dominates the cost.
    return IdealSim.from_mode("vampires_f760_10zern")


def test_reference_psf_shape_and_normalization(ideal):
    ref = ideal.reference_psf
    assert ref.shape == (128, 128)
    # per_sample_norm output is min-max normalized
    assert ref.min() == pytest.approx(0.0)
    assert ref.max() == pytest.approx(1.0)
    # Diffraction-limited reference peaks at the field center
    peak = np.unravel_index(ref.argmax(), ref.shape)
    assert abs(peak[0] - 64) <= 1 and abs(peak[1] - 64) <= 1


def test_actuation_changes_psf(ideal):
    acts = np.zeros(10)
    acts[3] = 0.5  # half-amplitude poke of one mid-order Zernike
    aberrated = ideal.psf(actuations={"zernike_dm": acts})
    assert aberrated.shape == (128, 128)
    assert not np.allclose(aberrated, ideal.reference_psf)


def test_sample_echoes_actuations(ideal):
    acts = np.zeros(10)
    result = ideal.sample(actuations={"zernike_dm": acts})
    assert "images" in result and "actuations" in result
    assert "zernike_dm" in result["actuations"]
