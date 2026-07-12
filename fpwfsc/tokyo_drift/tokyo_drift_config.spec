[MODE]
    mode name           = string
    calibration profile = string_or_none(default=None)

[LOOP_SETTINGS]
    Plot              = boolean(default=True)
    N iter            = integer(min=1)
    gain              = float(min=0)
    leak factor       = float(min=0, max=1)
    strehl early stop = float_or_none(default=None)
    predictor         = option('model', 'oracle', 'random_walk', default='model')
    strehl method     = option('vandam', 'proxy', default='vandam')

[MODEL]
    initial move sigma = float(min=0, default=0.01)
    device             = string(default='cpu')

[DM]
    dm channel               = string(default='dm00disp04')
    max peak to valley (um)  = float(min=0)
    max actuator stroke (um) = float(min=0)

# Model<->instrument alignment (GUI display name). 'bench sim preset' is
# a sim-only control (hidden on real hardware); 'probe amplitude' drives
# the calibration probe poke in both sim and real modes.
[ALIGNMENT]
    bench sim preset = string(default='easy')
    probe amplitude  = float(min=0, default=0.3)

# Displayed as "Noise mitigation". Two SNR levers: simulated source
# brightness (sim-only) and frame averaging (all backends; the camera's
# own capture averaging, no pipeline-side infrastructure). Both the
# loop and the calibration probes respect these. Coronagraphic modes
# read a faint speckle field and need materially more SNR than the
# bright no-coro core; the default averaging is chosen so calibration
# and the coro loop both work at the training-era flux.
[SNR]
    int phot flux exponent = float(default=3.5)
    frames to average      = integer(min=1, default=8)

# Displayed as "Test WFE injection params". Two seeds, two owners:
# 'seed' pins the BENCH (misalignment truth + detector noise) so a
# saved calibration stays valid across runs; 'wfe seed' pins the
# injected-WFE episode (error draw + the model's initial diversity
# move). The None default gives a fresh injected error every run
# against the same bench; tests set an integer for reproducibility.
[SIMULATION]
    seed              = integer_or_none(default=None)
    wfe seed          = integer_or_none(default=None)
    initial error rms = float(min=0, default=0.15)

[CAMERA CALIBRATION]
    background file = string(default='')
    masterflat file = string(default='')
    badpix file     = string(default='')
    # With no background file, optionally synthesize one from the frame's
    # border rows/columns. Off by default: the NN was trained on
    # unsubtracted min-max-normalized frames, and on faint coronagraphic
    # frames the border estimate subtracts real halo flux and injects
    # correlated stripe noise.
    estimate background from border = boolean(default=False)

[IO]
    save_log        = boolean(default=False)
    log_path        = string(default='')
    hitchhiker mode = boolean(default=False)
    hitchhiker path = string(default='')
