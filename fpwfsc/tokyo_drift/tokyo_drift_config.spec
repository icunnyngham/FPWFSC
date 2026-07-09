[MODE]
    mode name           = string
    calibration profile = string_or_none(default=None)

[LOOP_SETTINGS]
    Plot              = boolean(default=True)
    N iter            = integer(min=1)
    gain              = float(min=0)
    leak factor       = float(min=0, max=1)
    strehl early stop = float_or_none(default=None)
    predictor         = option('oracle', 'random_walk', 'model', default='oracle')
    strehl method     = option('vandam', 'proxy', default='vandam')

[MODEL]
    initial move sigma = float(min=0, default=0.01)
    device             = string(default='cpu')

[DM]
    dm channel               = string(default='dm00disp04')
    max peak to valley (um)  = float(min=0)
    max actuator stroke (um) = float(min=0)

[SIMULATION]
    bench sim preset  = string(default='easy')
    seed              = integer_or_none(default=None)
    initial error rms = float(min=0, default=0.15)

[CALIBRATION]
    probe amplitude = float(min=0, default=0.3)

[CAMERA CALIBRATION]
    background file = string(default='')
    masterflat file = string(default='')
    badpix file     = string(default='')

[IO]
    save_log        = boolean(default=False)
    log_path        = string(default='')
    hitchhiker mode = boolean(default=False)
    hitchhiker path = string(default='')
