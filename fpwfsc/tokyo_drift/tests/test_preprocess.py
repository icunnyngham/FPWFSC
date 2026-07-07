"""Preprocessing tests: rotate-then-align recovers synthetic mangling."""
import numpy as np
import pytest
from scipy.ndimage import rotate as nd_rotate

from fpwfsc.tokyo_drift.preprocess import PreprocessImage


def _gauss_blob(size, row, col, sigma=2.5):
    yy, xx = np.mgrid[:size, :size]
    return np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma ** 2))


def test_find_brightest_center():
    proc = PreprocessImage(crop_res=32)
    proc.find_brightest_center(_gauss_blob(256, 100, 180))
    assert (proc.cen_y, proc.cen_x) == (100, 180)


def test_process_crops_around_brightest():
    frame = _gauss_blob(256, 90, 170)
    proc = PreprocessImage(crop_res=64, rot_angle=0.0)
    out = proc.process(frame)
    assert out.shape == (64, 64)
    # Blob centered in the crop
    peak = np.unravel_index(out.argmax(), out.shape)
    assert abs(peak[0] - 32) <= 1 and abs(peak[1] - 32) <= 1
    # Normalized to [0, 1]
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)


def test_process_undoes_camera_rotation():
    """A frame rotated by +theta, preprocessed with rot_angle=-theta,
    recovers the unrotated scene (two-blob asymmetric pattern)."""
    scene = (_gauss_blob(300, 150, 150)
             + 0.5 * _gauss_blob(300, 150, 190))  # companion along +x
    camera_frame = nd_rotate(scene, 41.0, order=1, reshape=False)

    proc = PreprocessImage(crop_res=96, rot_angle=-41.0)
    out = proc.process(camera_frame, normalize=False)

    # Primary at crop center, companion restored along +x (same row)
    primary = np.unravel_index(out.argmax(), out.shape)
    masked = out.copy()
    masked[primary[0] - 8:primary[0] + 8, primary[1] - 8:primary[1] + 8] = 0
    companion = np.unravel_index(masked.argmax(), masked.shape)
    assert abs(companion[0] - primary[0]) <= 2       # same row
    assert 35 <= companion[1] - primary[1] <= 45     # ~40 px along +x


def test_find_conv_center_refines_toward_true_center():
    """With a deliberately wrong stored center, one conv-center pass on
    the processed crop moves the estimate to the true blob location."""
    frame = _gauss_blob(256, 128, 128)
    ref = _gauss_blob(64, 32, 32)

    proc = PreprocessImage(crop_res=64, rot_angle=0.0,
                           center_x=118, center_y=133)  # off by (-10, +5)
    crop = proc.process(frame, normalize=False)
    proc.find_conv_center(crop, ref)
    # Even-sized 'same' convolution has a half-pixel-ambiguous center,
    # so the bench method is quantized to +-1 px (it ran with even 128^2
    # crops on the bench too).
    assert abs(proc.cen_y - 128) <= 1 and abs(proc.cen_x - 128) <= 1


def test_edge_center_crop_zero_pads_to_size():
    """A center near the frame edge must still yield a full-size crop
    (zero-padded), never a silently mis-shaped array."""
    frame = _gauss_blob(256, 10, 250)  # near the top-right corner
    proc = PreprocessImage(crop_res=64, rot_angle=0.0)
    out = proc.process(frame, normalize=False)
    assert out.shape == (64, 64)
    peak = np.unravel_index(out.argmax(), out.shape)
    assert abs(peak[0] - 32) <= 1 and abs(peak[1] - 32) <= 1


def test_flips_applied_after_crop():
    frame = _gauss_blob(256, 128, 128) + 0.5 * _gauss_blob(256, 128, 148)
    base = PreprocessImage(crop_res=64, center_x=128, center_y=128)
    flipped = PreprocessImage(crop_res=64, center_x=128, center_y=128,
                              flip_horizontal=True)
    np.testing.assert_array_equal(flipped.process(frame),
                                  np.fliplr(base.process(frame)))
