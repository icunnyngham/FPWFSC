"""Model-native ideal simulator.

A thin wrapper around a telescope-sim (TS2) pipeline built from a
mode's ``ts2_config.yaml`` — the exact optical configuration the
mode's NN was trained against. Used as the algorithm's reference PSF
source and as the comparison target for calibration fits.

This is deliberately *not* the hardware stand-in: the misaligned
"bench sim" that plays the camera/DM in sim mode is a separate,
derived configuration.
"""
import numpy as np


class IdealSim:
    """Direct TS2 sample of a mode's training configuration.

    Parameters
    ----------
    ts2_config
        Path to a telescope-sim YAML config.
    """

    def __init__(self, ts2_config):
        try:
            from telescope_sim import TelescopeSim
        except ImportError as exc:
            raise ImportError(
                "telescope-sim is required for tokyo_drift simulation; "
                "install with: pip install 'fpwfsc[tokyo-drift]'") from exc
        self.config_path = str(ts2_config)
        self.sim = TelescopeSim.from_yaml(self.config_path)
        # Sampled at construction, before any actuation has touched the
        # (stateful) correctors, so this is the true flat-DM reference.
        self._reference_psf = self.psf()

    @classmethod
    def from_mode(cls, mode_name):
        """Build from a registered mode name (see ``mode_registry``)."""
        from ..mode_registry import ts2_config_path
        return cls(ts2_config_path(mode_name))

    def sample(self, actuations=None, **kwargs):
        """Raw TS2 sample: returns the full ``{'images': ..., 'actuations':
        ..., 'strehls': ...}`` dict."""
        return self.sim.sample(actuations=actuations, **kwargs)

    def psf(self, actuations=None, output="psf"):
        """Sample and return one output image as a squeezed 2-D array."""
        images = self.sample(actuations=actuations)["images"][output]
        return np.squeeze(np.asarray(images))

    @property
    def reference_psf(self):
        """Flat-DM PSF (post-processed), captured at construction."""
        return self._reference_psf
