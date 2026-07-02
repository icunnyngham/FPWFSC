[MODE]
    mode name           = string
    calibration profile = string_or_none(default=None)

[LOOP_SETTINGS]
    Plot              = boolean(default=True)
    N iter            = integer(min=1)
    gain              = float(min=0)
    leak factor       = float(min=0, max=1)
    strehl early stop = float_or_none(default=None)
    predictor         = option('oracle', 'random_walk', default='oracle')

[DM]
    dm channel               = string(default='dm00disp04')
    max peak to valley (um)  = float(min=0)
    max actuator stroke (um) = float(min=0)

[SIMULATION]
    bench sim preset  = string(default='easy')
    seed              = integer_or_none(default=None)
    initial error rms = float(min=0, default=0.15)

[IO]
    save_log = boolean(default=False)
    log_path = string(default='')
