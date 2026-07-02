"""Detector-frame preprocessing: rotate -> find center -> crop -> normalize.

Ported from the SCExAO bench sessions' ``PreprocessImage``. The
ordering is the load-bearing design decision: at the large camera
rotations seen on VAMPIRES (~235 deg), alignment must happen *after*
rotation — ``common.support_functions.reduce_images`` aligns first and
is only valid for small (<~10 deg) misalignments, so tokyo_drift keeps
its own preprocessing.

Center finding offers the two bench strategies:

- ``find_brightest_center`` — argmax of the frame (Oct 2023; fails on
  multi-lobed aberrated PSFs).
- ``find_conv_center`` — argmax of the convolution with a reference
  (simulated) PSF (Mar 2024 onward; robust at high aberration). This is
  also the primitive the rotation-calibration sweep maximizes.
"""
import numpy as np
from scipy.ndimage import rotate
from scipy.signal import fftconvolve


class PreprocessImage:
    """Rotate, center, crop, and normalize a raw detector frame.

    Parameters come from the calibration profile: ``rot_angle`` is the
    camera-vs-model rotation, ``center_x``/``center_y`` the PSF center
    *in the rotated frame* (found automatically on first use if not
    given), flips applied after cropping.
    """

    def __init__(
        self,
        crop_res=128,
        rot_angle=0.0,
        center_x=None,
        center_y=None,
        flip_vertical=False,
        flip_horizontal=False,
        verbose=True,
    ):
        self.res = int(crop_res)
        self.rot_angle = float(rot_angle)
        self.cen_x = center_x
        self.cen_y = center_y
        self.flip_v = bool(flip_vertical)
        self.flip_h = bool(flip_horizontal)
        self.verbose = bool(verbose)

    def find_brightest_center(self, im):
        """Set the stored center to the brightest finite pixel."""
        fin_mask = np.isfinite(im)
        im_max = np.max(im[fin_mask])
        self.cen_y, self.cen_x = np.argwhere(im == im_max)[0]
        if self.verbose:
            print(f"New center set to x: {self.cen_x}, y: {self.cen_y}")

    def find_conv_center(self, im, ref_im):
        """Refine the stored center by cross-correlating with a
        reference PSF.

        ``im`` is a frame processed with the *current* center estimate;
        the correlation peak's offset from the frame center is added to
        the stored center.

        Deviation from the bench port: the bench used ``convolve2d``,
        which flips the template — identical for centro-symmetric
        references but wrong for asymmetric ones (e.g. the calibration
        probe PSF). FFT cross-correlation is the correct form and much
        faster.
        """
        conv_im = fftconvolve(im, ref_im[::-1, ::-1], mode="same")
        fin_mask = np.isfinite(conv_im)
        im_max = np.max(conv_im[fin_mask])
        conv_y, conv_x = np.argwhere(conv_im == im_max)[0]
        ce_y, ce_x = int(im.shape[0] / 2), int(im.shape[1] / 2)
        self.cen_y += conv_y - ce_y
        self.cen_x += conv_x - ce_x
        if self.verbose:
            print(f"New center set to x: {self.cen_x}, y: {self.cen_y}")

    def process(self, in_im, normalize=True):
        """Rotate -> (auto-)center -> crop -> flip -> normalize."""
        im = in_im.astype(np.float32)

        im = rotate(im, self.rot_angle, order=0)

        if self.cen_y is None or self.cen_x is None:
            self.find_brightest_center(im)

        h_res = int(self.res / 2)
        y, x = self.cen_y, self.cen_x
        im = im[y - h_res:y + h_res, x - h_res:x + h_res]

        if self.flip_v:
            im = np.flipud(im)
        if self.flip_h:
            im = np.fliplr(im)

        if normalize:
            im = (im - im.min()) / np.ptp(im)

        return im
