# PyTorch inference must coexist with hcipy/mkl in the same process; both
# vendor an OpenMP runtime and libomp aborts on the double-load. Allow it
# (CPU inference, no measurable numerical impact) before torch is imported.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time

import numpy as np

from .model_torch import FFModelTorch

# Architecture keys carried in the self-describing checkpoint's ``meta``.
# ``input_hw`` is absent in the earliest checkpoints (defaults to 128) and
# 120 for the randcrop VVC model; passed through when present.
_ARCH_KEYS = ("n_modes", "conv_channels", "conv_kernel", "n_conv",
              "dense_size", "n_dense", "psf_channels", "input_hw")


class TorchPredictor:
    """Trained Tokyo Drift NN as a loop predictor.

    Implements the same ``predict(frames, actuation) -> ndarray`` contract
    as the dummy predictors (:mod:`fpwfsc.tokyo_drift.predictors`), so the
    closed loop is agnostic to what drives it. Wraps the vendored
    :class:`~fpwfsc.tokyo_drift.model_torch.FFModelTorch` (the byte-identical
    PyTorch twin of the deployed Keras model), which is *self-describing*:
    the ``.pt`` carries the full architecture in its ``meta`` block, so
    nothing about a specific model is hardcoded here.

    Two sign conventions bridge the loop and the model; both are exercised
    by the sign-validation test against the training-matched ideal sim:

    - **Return** is the estimated *current* wavefront error in the training
      modal basis; the loop applies ``state = leak*state - gain*prediction``
      (do not negate here).
    - **Actuation input** — the loop reports ``delta_actuation`` as the DM
      *command* change between the two frames (``state_k - state_{k-1}``).
      The model was trained on the *aberration* change in the opposite
      convention (``state_before - state_after``). Since the net wavefront
      is ``external_error + command`` (the calibrated DM adds +command; the
      loop's convergence is the empirical proof), the aberration change is
      the command change, and the training convention is its negation. So
      the model move is ``-delta_actuation``.
    """

    def __init__(self, checkpoint_path, device=None):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for the model predictor; install with: "
                "pip install 'fpwfsc[tokyo-drift]'") from exc

        blob = torch.load(str(checkpoint_path), map_location="cpu")
        meta = blob.get("meta", {})
        arch = {k: meta[k] for k in _ARCH_KEYS if k in meta}

        self.device = device or "cpu"
        self.model = FFModelTorch(**arch)
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.n_modes = int(meta.get("n_modes", self.model.n_modes))
        self.run_id = meta.get("run_id")
        self.last_latency_s = None
        self._printed_latency = False

    @classmethod
    def from_mode(cls, mode_name, device=None):
        """Build from a registered mode, resolving the manifest checkpoint."""
        from .mode_registry import checkpoint_path
        return cls(checkpoint_path(mode_name), device=device)

    def predict(self, frames, actuation):
        """Estimate the current modal wavefront error.

        Parameters
        ----------
        frames
            ``(2, res, res)`` stack of the previous and current normalized
            PSF frames (``frames[0]`` = before, ``frames[1]`` = after).
        actuation
            The loop's ``delta_actuation`` — the DM command change between
            the two frames. Negated internally to the model's training
            convention (see the class docstring).
        """
        frames = np.asarray(frames)
        psf_before, psf_after = frames[0], frames[1]
        move = -np.asarray(actuation, dtype=float)

        t0 = time.perf_counter()
        residual = self.model.predict_residual(
            psf_before, psf_after, move, device=self.device)
        self.last_latency_s = time.perf_counter() - t0

        if not self._printed_latency:
            print(f"tokyo_drift: model inference latency "
                  f"{self.last_latency_s * 1e3:.1f} ms "
                  f"(device={self.device})")
            self._printed_latency = True
        return np.asarray(residual, dtype=float)
