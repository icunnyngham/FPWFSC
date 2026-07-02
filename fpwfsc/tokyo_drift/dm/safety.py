"""DM command safety bounds.

Refuses commands that exceed configurable peak-to-valley or
per-actuator stroke limits (a mispredicting NN must not be able to
damage the DM during commissioning), and warns loudly when a command
approaches them.
"""
import numpy as np


class DMSafetyError(ValueError):
    """A DM command exceeded the configured safety bounds."""


class DMSafetyBounds:
    """Validate DM commands (in microns) against stroke limits.

    Parameters
    ----------
    max_ptv_um
        Maximum allowed surface peak-to-valley, microns.
    max_stroke_um
        Maximum allowed absolute single-actuator stroke, microns.
    warn_fraction
        Print a warning when a command exceeds this fraction of either
        bound (warn, don't clip — silent clipping hides diverging loops).
    """

    def __init__(self, max_ptv_um, max_stroke_um, warn_fraction=0.8):
        self.max_ptv_um = float(max_ptv_um)
        self.max_stroke_um = float(max_stroke_um)
        self.warn_fraction = float(warn_fraction)

    def check(self, command_um):
        """Return the command unchanged if safe; raise DMSafetyError if
        it exceeds either bound."""
        cmd = np.asarray(command_um, dtype=float)
        ptv = float(np.ptp(cmd))
        stroke = float(np.max(np.abs(cmd)))

        if ptv > self.max_ptv_um or stroke > self.max_stroke_um:
            raise DMSafetyError(
                f"DM command refused: peak-to-valley {ptv:.3f} um "
                f"(limit {self.max_ptv_um}), max stroke {stroke:.3f} um "
                f"(limit {self.max_stroke_um})")
        if (ptv > self.warn_fraction * self.max_ptv_um
                or stroke > self.warn_fraction * self.max_stroke_um):
            print(f"WARNING: DM command near limits - peak-to-valley "
                  f"{ptv:.3f}/{self.max_ptv_um} um, "
                  f"stroke {stroke:.3f}/{self.max_stroke_um} um")
        return command_um
