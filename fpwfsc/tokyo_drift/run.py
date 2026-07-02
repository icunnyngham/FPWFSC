#!/usr/bin/env python
"""Tokyo Drift — NN-driven focal-plane wavefront sensing & control loop.

The Tokyo Drift method predicts the wavefront error from two consecutive
focal-plane frames plus the DM actuation applied between them (an NN
analogue of Fast & Furious), then corrects with a leaky integrator.

This module is the pipeline entry point, following the same contract as
the other FPWFSC pipelines: ``run()`` reads a validated .ini config and
executes the control loop; the GUI, the command line, and notebooks all
call this same function.
"""
import threading
from pathlib import Path

from ..common import support_functions as sf


def run(camera=None, aosystem=None, config=None, configspec=None,
        my_event=None, plotter=None):
    """Run the Tokyo Drift control loop.

    Parameters
    ----------
    camera, aosystem
        ``'Sim'`` string sentinels for simulation mode, or real-hardware
        wrapper instances (see ``gui_helper.load_instruments``).
    config, configspec
        Path (or dict) for the .ini config and its .spec validator.
    my_event
        ``threading.Event`` used by the GUI to signal stop.
    plotter
        Optional live plotter; must expose an ``update(...)`` method.

    Returns
    -------
    The validated settings object.
    """
    if my_event is None:
        my_event = threading.Event()
    settings = sf.validate_config(config, configspec)

    mode_name = settings['MODE']['mode name']
    n_iter = settings['LOOP_SETTINGS']['N iter']

    if camera == 'Sim' and aosystem == 'Sim':
        pass  # simulator backends are constructed here
    else:
        raise NotImplementedError(
            "tokyo_drift real-hardware backends are not implemented yet; "
            "run with camera='Sim', aosystem='Sim'")

    if my_event.is_set():
        return settings

    print(f"tokyo_drift: config OK - mode '{mode_name}', "
          f"{n_iter} iterations requested.")
    print("tokyo_drift: control loop not implemented yet; exiting.")
    return settings


if __name__ == "__main__":
    camera = "Sim"
    aosystem = "Sim"

    script_dir = Path(__file__).parent
    config_path = script_dir / "tokyo_drift_config_sim.ini"
    spec_path = script_dir / "tokyo_drift_config.spec"
    # When the live plotter exists, construct it here and pass it through.
    run(camera, aosystem, config=str(config_path),
        configspec=str(spec_path), plotter=None)
