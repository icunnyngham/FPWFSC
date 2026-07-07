"""Bench simulator: the misaligned hardware stand-in for sim mode.

Where :class:`~fpwfsc.tokyo_drift.sim.ideal_sim.IdealSim` is the clean
model-native configuration an NN was trained against, the bench sim
plays the *hardware*: the same optical train, deliberately mangled the
way a real instrument mounting mangles it. It exposes the hardware
interface (``take_image`` / ``set_dm_data`` with a raw 50x50-microns
command), so everything above that boundary — preprocessing,
calibration, the NN loop — runs identical code in sim and on the bench,
and calibration routines can be validated against *known* injected
ground truth.

The mangling has two halves:

- **DM side** (inside the optical propagation): the modal training DM is
  replaced by a telescope-sim ``actuator_grid`` DM (50x50
  influence-function actuators) with injected rotation, flips, and
  actuation-scale error baked into its geometry.
- **Image side** (after propagation): the focal plane is rendered
  noiseless at higher resolution and larger field than the training
  frame, then rotated, cropped off-center to the detector frame, flux
  scaled, and *only then* given photon + read noise — noise is applied
  last so resampling never distorts its statistics.

Injected parameters are drawn from a named *preset* of priors
(``bench_sim_presets/<name>.yaml``) and recorded in ``self.truth`` for
calibration-recovery validation.
"""
import copy
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage

PRESETS_DIR = Path(__file__).resolve().parents[1] / "bench_sim_presets"

# SCExAO 2k DM geometry: 50x50 actuators at 0.17 m pupil-projected pitch.
DM_NUM_ACTUATORS = 50
DM_PITCH_M = 0.17
# Commands are in microns of surface (the bench SHM convention); the
# nominal command-to-meters factor an *un*-miscalibrated DM would have.
DM_NOMINAL_SCALE = 1.0e-6
# Minimum pupil-grid extent that contains the full 50 x 0.17 m actuator
# lattice (influence functions outside the grid are silently truncated).
MIN_PUPIL_EXTENT = 8.65


def list_presets(presets_dir=PRESETS_DIR):
    """Names of the available mangling-prior presets."""
    return sorted(p.stem for p in Path(presets_dir).glob("*.yaml"))


def load_preset(name, presets_dir=PRESETS_DIR):
    """Load a mangling-prior preset by name."""
    import yaml
    path = Path(presets_dir) / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in Path(presets_dir).glob("*.yaml"))
        raise KeyError(f"unknown bench-sim preset {name!r}; available: {available}")
    with open(path) as f:
        return yaml.safe_load(f)


def sample_truth(preset, rng):
    """Draw one set of injected misalignment parameters from a preset.

    Returns a dict with keys: ``image_rot_deg``, ``dm_rot_deg``,
    ``crop_dx``, ``crop_dy``, ``dm_scale``, ``dm_flip_x``, ``dm_flip_y``.
    """
    def uniform(block):
        lo, hi = float(block["min"]), float(block["max"])
        return lo if lo == hi else rng.uniform(lo, hi)

    crop = preset["crop_offset_px"]
    lo, hi = int(crop["min"]), int(crop["max"])
    return {
        "image_rot_deg": uniform(preset["image_rotation_deg"]),
        "dm_rot_deg": uniform(preset["dm_rotation_deg"]),
        "crop_dx": int(rng.integers(lo, hi + 1)),
        "crop_dy": int(rng.integers(lo, hi + 1)),
        "dm_scale": uniform(preset["dm_scale"]),
        "dm_flip_x": bool(rng.random() < float(preset["dm_flip_x_probability"])),
        "dm_flip_y": bool(rng.random() < float(preset["dm_flip_y_probability"])),
    }


def derive_bench_config(base_config, truth, render_res=512,
                        min_pupil_extent=MIN_PUPIL_EXTENT,
                        num_actuators=DM_NUM_ACTUATORS,
                        actuator_pitch=DM_PITCH_M):
    """Derive the bench-sim TS2 config from a mode's ideal config.

    The pupil/filter/bandwidth axes are preserved exactly; the focal
    plane is re-rendered at ``render_res`` with the *same* angular pixel
    scale; the modal corrector is swapped for a misaligned
    ``actuator_grid`` DM; output post-processing is stripped (the bench
    sim produces raw intensity — flux scaling and noise happen in the
    mangling layer, after resampling).
    """
    cfg = copy.deepcopy(base_config)

    cfg["pupil"]["extent"] = max(float(cfg["pupil"]["extent"]),
                                 float(min_pupil_extent))

    for fp in cfg["focal_planes"].values():
        mas_per_pix = float(fp["focal_extent"]) / int(fp["focal_res"])
        fp["focal_res"] = int(render_res)
        fp["focal_extent"] = mas_per_pix * int(render_res)

    cfg["correctors"] = {
        "bench_dm": {
            "type": "actuator_grid",
            "num_actuators": int(num_actuators),
            "actuator_pitch": float(actuator_pitch),
            "influence": "gaussian",
            "crosstalk": 0.15,
            "rotation_deg": float(truth["dm_rot_deg"]),
            "flip_x": bool(truth["dm_flip_x"]),
            "flip_y": bool(truth["dm_flip_y"]),
            "actuate_scale": DM_NOMINAL_SCALE * float(truth["dm_scale"]),
            "wavefront_role": "actuate",
            "target_strategy": "none",
        }
    }
    cfg["corrector_chain"] = ["bench_dm"]

    for output in cfg["outputs"].values():
        output.pop("post_processing", None)

    return cfg


def mangle_frame(image, image_rot_deg, crop_dx, crop_dy, frame_size,
                 rng=None, total_photons=None, read_noise=0.0):
    """Rotate -> off-center crop -> flux scale -> noise (strictly last).

    Parameters
    ----------
    image
        Noiseless rendered intensity (2-D, larger than ``frame_size``).
    image_rot_deg
        Camera rotation relative to the model, degrees.
    crop_dx, crop_dy
        Detector-frame center offset from the rendered-field center,
        pixels (x = columns, y = rows).
    frame_size
        Output frame is ``frame_size x frame_size``.
    rng
        ``numpy.random.Generator`` for the noise draws; ``None`` returns
        the deterministic noiseless frame (flux-scaled if requested).
    total_photons
        If given, the *full rotated field* is scaled to this many
        photons before cropping (light falling outside the crop is
        lost, as on a real detector).
    read_noise
        Gaussian read noise sigma, photoelectrons.
    """
    image = np.asarray(image, dtype=float)
    rotated = ndimage.rotate(image, float(image_rot_deg),
                             reshape=False, order=1)
    np.clip(rotated, 0.0, None, out=rotated)

    if total_photons is not None and rotated.sum() > 0:
        rotated *= float(total_photons) / rotated.sum()

    h, w = rotated.shape
    row0 = h // 2 + int(crop_dy) - frame_size // 2
    col0 = w // 2 + int(crop_dx) - frame_size // 2
    if row0 < 0 or col0 < 0 or row0 + frame_size > h or col0 + frame_size > w:
        raise ValueError(
            f"crop ({frame_size}^2 at offset ({crop_dx}, {crop_dy})) falls "
            f"outside the rendered {h}x{w} field; increase render_res or "
            "shrink the crop-offset prior")
    frame = rotated[row0:row0 + frame_size, col0:col0 + frame_size]

    if rng is not None:
        frame = rng.poisson(frame).astype(float)
        if read_noise > 0:
            frame = frame + rng.normal(0.0, float(read_noise), frame.shape)
    return frame


class BenchSim:
    """Misaligned TS2 stand-in for the real camera + DM.

    Parameters
    ----------
    ts2_config
        Path to the mode's ideal TS2 YAML (the bench config is derived
        from it — never hand-maintained separately).
    preset
        Name of a mangling-prior preset (or a preset dict).
    seed
        Seed for both the truth-parameter draw and the per-frame noise.
    detector_frame
        Simulated detector frame size, pixels (must exceed the training
        frame so the true field position must be *discovered*).
    render_res
        Noiseless rendering resolution before mangling.
    int_phot_flux
        Photons / m^2 per integration (legacy MAS flux convention;
        multiplied by the aperture area from the config).
    read_noise
        Detector read noise sigma, photoelectrons.
    """

    def __init__(self, ts2_config, preset="easy", seed=None,
                 detector_frame=256, render_res=512,
                 int_phot_flux=3162.0, read_noise=5.0):
        try:
            from telescope_sim import TelescopeSim
        except ImportError as exc:
            raise ImportError(
                "telescope-sim is required for tokyo_drift simulation; "
                "install with: pip install 'fpwfsc[tokyo-drift]'") from exc
        import yaml

        self.rng = np.random.default_rng(seed)
        self.preset = load_preset(preset) if isinstance(preset, str) else dict(preset)
        self.truth = sample_truth(self.preset, self.rng)

        with open(ts2_config) as f:
            base_config = yaml.safe_load(f)
        self.derived_config = derive_bench_config(base_config, self.truth,
                                                  render_res=render_res)
        self.aperture_area = float(base_config["aperture"].get("area", 0.0))
        # The mode's training corrector block (kept for expressing
        # external errors in the training modal basis).
        self._base_corrector = base_config["correctors"][
            base_config["corrector_chain"][0]]

        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False)
        try:
            yaml.safe_dump(self.derived_config, tmp)
            tmp.close()
            self.sim = TelescopeSim.from_yaml(tmp.name)
        finally:
            os.unlink(tmp.name)

        self.detector_frame = int(detector_frame)
        self.int_phot_flux = float(int_phot_flux)
        self.read_noise = float(read_noise)
        n = DM_NUM_ACTUATORS
        self._dm_command = np.zeros((n, n))
        self._error_command = np.zeros((n, n))
        self._error_opd = None

    @classmethod
    def from_mode(cls, mode_name, **kwargs):
        from ..mode_registry import ts2_config_path
        return cls(ts2_config_path(mode_name), **kwargs)

    # --- hardware interface (the sim/real boundary) --------------------

    def set_dm_data(self, dm_microns):
        """Accept a raw 50x50 command in microns of surface — exactly
        what ``SCEXAO.set_dm_data`` ships to the shared-memory stream."""
        self._dm_command = self._validated(dm_microns)

    def set_error_command(self, dm_microns):
        """Inject a hidden static surface error (microns), added to
        every user command — "the DM's flat isn't flat". DM-borne error
        class: it renders through the same influence functions as
        corrections do. Sim-only side channel."""
        self._error_command = self._validated(dm_microns)

    def set_error_opd(self, opd_field):
        """Inject a hidden static EXTERNAL wavefront error (OPD in
        meters over the pupil grid) — the NCPA-like error class the
        real bench actually fights. Applied via the TS2 atmosphere
        hook, so it does NOT pass through the DM's influence functions;
        cancelling it requires the DM's *effective* command-to-
        wavefront gain, which is exactly what the dm_scale calibration
        measures. ``None`` clears."""
        self._error_opd = opd_field

    def set_modal_error(self, coefficients,
                        opd_per_unit=2.0 * DM_NOMINAL_SCALE):
        """External error (:meth:`set_error_opd`) expressed in the
        mode's training Zernike basis. ``opd_per_unit`` matches the OPD
        a reflective-DM poke of one coefficient unit would imprint
        (2 x the nominal surface scale)."""
        import hcipy
        pupil = self.derived_config["pupil"]
        grid = hcipy.make_pupil_grid(int(pupil["resolution"]),
                                     float(pupil["extent"]))
        corr = self._base_corrector
        basis = hcipy.make_zernike_basis(
            len(coefficients), float(corr["zernike_diameter"]), grid,
            starting_mode=int(corr.get("starting_mode", 2)))
        basis = hcipy.ModeBasis([b / np.max(np.abs(b)) for b in basis])
        opd = basis.linear_combination(np.asarray(coefficients, dtype=float))
        self.set_error_opd(opd * float(opd_per_unit))

    @staticmethod
    def _validated(dm_microns):
        cmd = np.asarray(dm_microns, dtype=float)
        n = DM_NUM_ACTUATORS
        if cmd.shape != (n, n):
            raise ValueError(f"expected ({n}, {n}) DM command, got {cmd.shape}")
        return cmd.copy()

    def _render(self):
        effective = self._dm_command + self._error_command
        kwargs = {}
        if self._error_opd is not None:
            opd = self._error_opd

            def atmos(wf):
                out = wf.copy()
                out.electric_field = out.electric_field * np.exp(
                    1j * wf.wavenumber * opd)
                return out

            kwargs["atmos"] = atmos
        return np.squeeze(np.asarray(
            self.sim.sample(actuations={"bench_dm": effective},
                            **kwargs)["images"]["psf"]))

    def take_image(self, average=1):
        """Render the current optical state and return a mangled,
        noisy detector frame (averaged over ``average`` noise draws)."""
        rendered = self._render()
        total_photons = (self.int_phot_flux * self.aperture_area
                         if self.aperture_area > 0 else None)
        frames = [
            mangle_frame(rendered,
                         self.truth["image_rot_deg"],
                         self.truth["crop_dx"], self.truth["crop_dy"],
                         self.detector_frame,
                         rng=self.rng,
                         total_photons=total_photons,
                         read_noise=self.read_noise)
            for _ in range(int(average))
        ]
        return frames[0] if len(frames) == 1 else np.mean(frames, axis=0)

    def take_image_noiseless(self):
        """Mangled but noise-free frame (diagnostics / calibration dev)."""
        rendered = self._render()
        return mangle_frame(rendered,
                            self.truth["image_rot_deg"],
                            self.truth["crop_dx"], self.truth["crop_dy"],
                            self.detector_frame, rng=None)


class BenchSimCamera:
    """Camera-side adapter (FPWFSC hardware duck type)."""

    def __init__(self, bench_sim):
        self.bench = bench_sim

    def take_image(self, average=1):
        return self.bench.take_image(average=average)


class BenchSimAO:
    """AO/DM-side adapter (FPWFSC hardware duck type)."""

    def __init__(self, bench_sim):
        self.bench = bench_sim

    def set_dm_data(self, dm_microns):
        self.bench.set_dm_data(dm_microns)
