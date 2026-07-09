"""Closed-loop machinery: leaky integrator + the control loop itself.

The loop structure mirrors the bench sessions (and fnf's loop): take an
image, preprocess it, predict the wavefront error from the last two
frames plus the actuation between them, integrate the correction, ship
it to the DM, repeat. The camera and DM enter only through
``take_image`` / ``send_command`` callables, so the identical loop runs
against the bench sim and the real hardware.
"""
import numpy as np


def peak_flux_ratio(frame):
    """Peak-to-total-flux ratio — the raw ingredient of the Strehl
    proxy (divide by a diffraction-limited reference's ratio)."""
    total = frame.sum()
    return float(frame.max() / total) if total > 0 else 0.0


def _normalize(frame):
    span = np.ptp(frame)
    return (frame - frame.min()) / span if span > 0 else np.zeros_like(frame)


class LeakyIntegrator:
    """``state = leak * state - gain * prediction``.

    The predictor estimates the current wavefront *error*, so the
    correction is its negative; ``leak < 1`` bleeds off accumulated
    state to bound integrator wind-up.
    """

    def __init__(self, n_modes, gain, leak=1.0):
        self.gain = float(gain)
        self.leak = float(leak)
        self.state = np.zeros(int(n_modes))

    def update(self, prediction):
        self.state = self.leak * self.state - self.gain * np.asarray(prediction)
        return self.state


def run_closed_loop(take_image, send_command, predictor, translator,
                    integrator, preprocess, n_iter, *,
                    safety=None, average=1, initial_move=None,
                    strehl_fn=None, strehl_early_stop=None,
                    stop_event=None, plotter=None, ideal_psf=None,
                    iteration_callback=None):
    """Run the NN closed loop.

    Parameters
    ----------
    take_image
        ``callable(average=N) -> raw detector frame``.
    send_command
        ``callable(cmd_um)`` shipping a 50x50 microns command to the DM.
    predictor
        Object with ``predict(frames, actuation) -> coefficients``.
    translator
        :class:`~fpwfsc.tokyo_drift.dm.TranslationDM` (or equivalent).
    integrator
        :class:`LeakyIntegrator`, created by the caller (the sim-mode
        oracle holds a reference to its state).
    preprocess
        :class:`~fpwfsc.tokyo_drift.preprocess.PreprocessImage`.
    n_iter
        Maximum iterations.
    safety
        Optional :class:`~fpwfsc.tokyo_drift.dm.DMSafetyBounds`;
        violations raise (fail loudly, never clip silently).
    initial_move
        Optional length-``n_modes`` diversity move applied to the DM
        *before* the first prediction, so the first frame pair spans a
        known actuation instead of two identical frames. Required by the
        NN predictor (temporal diversity disambiguates sign-degenerate
        modes); the oracle/random-walk predictors ignore actuation and
        pass ``None`` for the current, no-initial-move behavior. The move
        is commanded through the integrator/translator (and safety), so it
        persists and is corrected on top of, exactly as in the training
        eval loop.
    strehl_fn
        Optional ``callable(processed_unnormalized_frame) -> float``.
    strehl_early_stop
        Stop once the metric exceeds this value (None disables).
    stop_event
        ``threading.Event`` checked each iteration (GUI stop).
    plotter
        Optional live plotter taking dict payloads.
    ideal_psf
        Reference PSF forwarded to the plotter's "ideal" panel.
    iteration_callback
        Optional ``callable(payload_dict)`` invoked once per iteration
        after the command is sent, with keys ``iteration, raw,
        processed, prediction, state, command, strehl`` — the session
        logger's hook.

    Returns
    -------
    dict with ``strehls`` (NaN-padded history), ``states`` (per-iter
    integrator state), ``final_state``, ``iterations`` completed.
    """
    n_modes = translator.n_modes
    strehls = np.full(n_iter, np.nan)
    states = np.zeros((n_iter, n_modes))

    raw = take_image(average)
    prev_frame = _normalize(preprocess.process(raw, normalize=False))
    delta_actuation = np.zeros(n_modes)

    # Optional diversity move before the first prediction: command it,
    # seed the integrator state to match, and record it as the actuation
    # between prev_frame (pre-move) and the first in-loop frame (post-move).
    if initial_move is not None:
        move = np.asarray(initial_move, dtype=float).reshape(n_modes)
        integrator.state = move.copy()
        command = translator.command_microns(integrator.state)
        if safety is not None:
            safety.check(command)
        send_command(command)
        delta_actuation = move.copy()

    prev_state = integrator.state.copy()
    completed = 0

    for i in range(n_iter):
        if stop_event is not None and stop_event.is_set():
            break

        raw = take_image(average)
        processed = preprocess.process(raw, normalize=False)
        frame = _normalize(processed)
        if strehl_fn is not None:
            strehls[i] = strehl_fn(processed)

        prediction = predictor.predict(np.stack([prev_frame, frame]),
                                       delta_actuation)
        state = integrator.update(prediction)
        command = translator.command_microns(state)
        if safety is not None:
            safety.check(command)
        send_command(command)

        delta_actuation = state - prev_state
        prev_state = state.copy()
        prev_frame = frame
        states[i] = state
        completed = i + 1

        if iteration_callback is not None:
            iteration_callback({
                "iteration": i,
                "raw": raw,
                "processed": processed,
                "prediction": prediction,
                "state": state,
                "command": command,
                "strehl": strehls[i],
            })

        if strehl_fn is not None:
            print(f"tokyo_drift iter {i + 1}/{n_iter}: "
                  f"strehl_proxy={strehls[i]:.3f}")
        if plotter is not None:
            plotter.update({
                "n_iter": n_iter,
                "iteration": completed,
                "source": frame,
                "ideal": ideal_psf,
                "ideal_title": "Ideal (reference PSF)",
                "strehls": strehls,
                "mode_coeffs": state,
            })
        if (strehl_early_stop is not None and np.isfinite(strehls[i])
                and strehls[i] >= strehl_early_stop):
            print(f"tokyo_drift: early stop - strehl proxy "
                  f"{strehls[i]:.3f} >= {strehl_early_stop}")
            break

    return {
        "strehls": strehls,
        "states": states,
        "final_state": integrator.state.copy(),
        "iterations": completed,
    }


def manual_poke(coefficients, take_image, send_command, translator, *,
                preprocess=None, safety=None, average=1):
    """Debug utility: send one arbitrary modal command, take an image.

    Returns the (optionally preprocessed) frame so calibration UIs and
    notebooks can display command-vs-response side by side.
    """
    command = translator.command_microns(coefficients)
    if safety is not None:
        safety.check(command)
    send_command(command)
    raw = take_image(average)
    return preprocess.process(raw) if preprocess is not None else raw
