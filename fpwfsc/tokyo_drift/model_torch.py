"""
PyTorch twin of the Tokyo Drift 974j9jqt Keras model (FFModel, multi_output).

Original Keras architecture (verified from the .h5 model_config + summary):

    psfs        : (N, 128, 128, 2)  NHWC
      Conv2D(1024, k=5, s=2, padding='valid', relu)   128 -> 62
      Conv2D(1024, k=5, s=2, padding='valid', relu)    62 -> 29
      Conv2D(1024, k=5, s=2, padding='valid', relu)    29 -> 13
      Conv2D(1024, k=5, s=2, padding='valid', relu)    13 ->  5
      Flatten (channels-last)                          -> 25600
    actuation   : (N, 10)
      concatenate([flatten, actuation])                -> 25610
      Dense(2048, relu) x4
      final_mag        = Dense(10, linear)             regression (residual |Zernike|)
      final_sign_bool  = Dense(10, linear)             LOGITS (BCE-from-logits)

Conversion-critical details:
  * padding='valid' -> torch padding=0 (no asymmetric-pad complications).
  * Keras Flatten is channels-LAST row-major over (H, W, C). torch conv output is
    NCHW, so we permute to NHWC *before* flattening. This makes the flattened
    feature ordering identical to Keras, so the big dense_4 weight transposes
    directly with NO row reordering (the #1 silent conversion bug, avoided here).
  * concat order is [psf_features, actuation] (psf first), matching Keras.
  * The sign head outputs LOGITS -- apply sigmoid/threshold at inference, do not
    bake an activation into the module.

The nn.Module is NCHW-native (idiomatic torch). Feed psfs as (N, 2, 128, 128).
Use `predict_nhwc` to feed NHWC numpy arrays matching the original data pipeline.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFModelTorch(nn.Module):
    def __init__(self, n_modes: int = 10, conv_channels: int = 1024,
                 conv_kernel: int = 5, n_conv: int = 4,
                 dense_size: int = 2048, n_dense: int = 4,
                 psf_channels: int = 2):
        super().__init__()
        self.n_modes = n_modes

        # ---- conv branch (NCHW) ----
        convs = []
        in_ch = psf_channels
        for _ in range(n_conv):
            convs.append(nn.Conv2d(in_ch, conv_channels, conv_kernel,
                                   stride=2, padding=0))
            in_ch = conv_channels
        self.convs = nn.ModuleList(convs)

        # ---- dense branch ----
        # Flatten size depends on input H/W; for the canonical 128x128 input with
        # 4x (k=5,s=2,valid) it is 5*5*conv_channels = 25600. dense_4 input dim =
        # flatten(25600) + n_modes(10) = 25610.
        flat = self._infer_flat(psf_channels, conv_channels, conv_kernel, n_conv)
        self._flat_features = flat
        in_dim = flat + n_modes
        denses = []
        for _ in range(n_dense):
            denses.append(nn.Linear(in_dim, dense_size))
            in_dim = dense_size
        self.denses = nn.ModuleList(denses)

        self.final_mag = nn.Linear(dense_size, n_modes)
        self.final_sign_bool = nn.Linear(dense_size, n_modes)  # logits

    @staticmethod
    def _infer_flat(psf_channels, conv_channels, k, n_conv, hw=128):
        s = hw
        for _ in range(n_conv):
            s = (s - k) // 2 + 1  # valid padding, stride 2
        return s * s * conv_channels

    def forward(self, psfs: torch.Tensor, actuation: torch.Tensor):
        """psfs: (N, 2, 128, 128) NCHW ; actuation: (N, n_modes)."""
        x = psfs
        for conv in self.convs:
            x = F.relu(conv(x))
        # Keras Flatten is channels-last: permute NCHW -> NHWC, then flatten.
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        z = torch.cat([x, actuation], dim=1)  # psf features first, then actuation
        for dense in self.denses:
            z = F.relu(dense(z))
        mag = self.final_mag(z)
        sign_logits = self.final_sign_bool(z)
        return mag, sign_logits

    # ---- convenience wrappers -------------------------------------------------
    @torch.no_grad()
    def predict_nhwc(self, psfs_nhwc: np.ndarray, actuation: np.ndarray,
                     device: str = "cpu", dtype=torch.float32):
        """Accept NHWC psfs (N,128,128,2) as produced by the data pipeline and
        length-n_modes actuation; return (mag, sign_logits) as numpy arrays."""
        self.eval()
        psfs = torch.as_tensor(np.asarray(psfs_nhwc), dtype=dtype, device=device)
        if psfs.ndim == 3:
            psfs = psfs.unsqueeze(0)
        psfs = psfs.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        act = torch.as_tensor(np.asarray(actuation), dtype=dtype, device=device)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        mag, sign = self(psfs, act)
        return mag.cpu().numpy(), sign.cpu().numpy()

    def predict_residual(self, psf0, psf1, applied_nudge, device="cpu"):
        """Framework-agnostic eval-loop callable (Track C).

        psf0, psf1: (128,128,1) or (128,128) broadband frames (NHWC-style).
        applied_nudge: length-n_modes vector (the 'messy' actuation).
        Returns signed residual Zernike vector = mag * sign, where
        sign = +1 where sigmoid(logit) >= 0.5 else -1  (bool head convention:
        final_sign_bool trained on clip(sign(err_final),0,1), i.e. 1 => positive).
        """
        p0 = np.asarray(psf0, np.float32).reshape(128, 128, -1)
        p1 = np.asarray(psf1, np.float32).reshape(128, 128, -1)
        psfs = np.concatenate([p0, p1], axis=-1)  # (128,128,2)
        mag, logits = self.predict_nhwc(psfs[None], np.asarray(applied_nudge)[None],
                                        device=device)
        sign = np.where(1 / (1 + np.exp(-logits)) >= 0.5, 1.0, -1.0)
        return (mag * sign)[0]


def build_and_load(state_dict_path: str, map_location: str = "cpu", **kwargs):
    model = FFModelTorch(**kwargs)
    sd = torch.load(state_dict_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


if __name__ == "__main__":
    m = FFModelTorch()
    n_params = sum(p.numel() for p in m.parameters())
    print(m)
    print(f"\nflat_features = {m._flat_features}")
    print(f"total params  = {n_params:,}  (expect 143,779,860)")
