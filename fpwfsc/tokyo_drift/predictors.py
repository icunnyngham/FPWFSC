"""Wavefront-error predictors.

A predictor implements the interface the trained NN will implement:

    predict(frames, actuation) -> ndarray (n_modes,)

where ``frames`` is a (2, res, res) stack of the previous and current
normalized PSF frames and ``actuation`` is the mode-coefficient delta
applied between them — the temporal-diversity input that disambiguates
modes degenerate in a single image. The return value is the estimated
*current* wavefront error in mode coefficients (the correction is its
negative, applied through the leaky integrator).

Two dummy predictors stand in for the NN during development:

- :class:`RandomWalkPredictor` — pure noise. A loop driven by it must
  wander/diverge; if it converges, something is leaking information.
- :class:`CheatingOracle` — reads the true residual through a sim-only
  side channel and returns it with configurable fuzz. A loop driven by
  it must converge; this validates every part of the chain *except*
  the NN (preprocessing, translation, DM commanding, integrator).
"""
import numpy as np


class RandomWalkPredictor:
    """Returns random mode coefficients — the divergence sanity check."""

    def __init__(self, n_modes, step_sigma=0.1, seed=None, rng=None):
        self.n_modes = int(n_modes)
        self.step_sigma = float(step_sigma)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def predict(self, frames, actuation):
        return self.rng.normal(0.0, self.step_sigma, self.n_modes)


class CheatingOracle:
    """Returns the true residual (fuzzed) via a sim-only side channel.

    Parameters
    ----------
    residual_fn
        Callable returning the current true modal residual — typically
        ``lambda: error_coeffs + integrator.state`` wired by the sim
        run path. Nothing on the real bench can provide this.
    fuzz_sigma
        Multiplicative per-mode fuzz: the prediction is
        ``residual * (1 + N(0, fuzz_sigma))`` — always pointing in the
        known right direction for small fuzz.
    """

    def __init__(self, residual_fn, fuzz_sigma=0.05, seed=None, rng=None):
        self.residual_fn = residual_fn
        self.fuzz_sigma = float(fuzz_sigma)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def predict(self, frames, actuation):
        residual = np.asarray(self.residual_fn(), dtype=float)
        return residual * (1.0 + self.rng.normal(0.0, self.fuzz_sigma,
                                                 residual.shape))
