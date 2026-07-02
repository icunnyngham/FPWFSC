"""
Simulate the SCExAO pupil
"""
from argparse import ArgumentParser

import hcipy as hp
import numpy as np
from astropy.io import fits

__all__ = ["generate_pupil"]

## constants
PUPIL_DIAMETER = 7.92  # m
OBSTRUCTION_DIAMETER = 2.403  # m
INNER_RATIO = OBSTRUCTION_DIAMETER / PUPIL_DIAMETER
SPIDER_WIDTH = 0.1735  # m
SPIDER_OFFSET = 0.639  # m, spider intersection offset
SPIDER_ANGLE = 51.75  # deg
# PUPIL_ANGLE = -39  # deg
ACTUATOR_SPIDER_WIDTH = 0.089  # m
ACTUATOR_SPIDER_OFFSET = (0.521, -1.045)
ACTUATOR_DIAMETER = 0.632  # m
ACTUATOR_OFFSET = ((1.765, 1.431), (-0.498, -2.331))  # (x, y), m  # (x, y), m

## command-line arg parsing
parser = ArgumentParser()
parser.add_argument("filename", help="output FITS filename")
parser.add_argument("-n",
                    type=int,
                    default=256,
                    help="size of pixel grid. Defaults to %(default)d")
parser.add_argument(
    "-o",
    "--outer",
    default=1,
    type=float,
    help=
    "Outer diameter of pupil as a fraction of the true pupil diameter. Defaults to %(default).02f",
)
parser.add_argument(
    "-i",
    "--inner",
    default=INNER_RATIO,
    type=float,
    help=
    "Inner diameter of pupil as a fraction of the true pupil diameter. Defaults to %(default).02f",
)
parser.add_argument(
    "-s",
    "--scale",
    default=1,
    type=float,
    help=
    "Scale factor for spiders and bad DM actuator masks as a fraction of their true size. Defaults to %(default).02f",
)
parser.add_argument(
    "-r",
    "--rotate",
    default=0,
    type=float,
    help="Pupil offset angle in degrees. Defaults to %(default).01f",
)
parser.add_argument(
    "-f",
    "--oversample-factor",
    default=8,
    type=int,
    help=
    "Oversample factor for supersampled evaluation of the pupil grid. Defaults to %(default)d",
)
parser.add_argument(
    "--no-spiders",
    action="store_false",
    dest="spiders",
    help="Don't create spider obstructions.",
)
parser.add_argument(
    "--no-actuators",
    action="store_false",
    dest="actuators",
    help="Don't create bad actuator mask obstructions.",
)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def field_combine(field1, field2):
    return lambda grid: field1(grid) * field2(grid)


def generate_pupil(
    n: int = 256,
    outer: float = 1,
    inner: float = INNER_RATIO,
    scale: float = 1,
    angle: float = 0,
    oversample: int = 8,
    spiders: bool = True,
    actuators: bool = True,
    pupil_grid = None,
):
    f"""
    Generate a SCExAO pupil parametrically.

    Parameters
    ----------
    n : int, optional
        Grid size in pixels. Default is 256
    outer : float, optional
        Outer pupil diameter as a fraction of the true diameter. Default is 1.0
    inner : float, optional
        Diameter of central obstruction as a fraction of the true diameter. Default is {INNER_RATIO:.03f}
    scale : float, optional
        Scale factor for over-sizing spiders and actuator masks. Default is 1.0
    angle : float, optional
        Pupil rotation angle, in degrees. Default is 0
    oversample : int, optional
        Oversample factor for supersampling the pupil grid. Default is 8
    spiders : bool, optional
        Add spiders to pupil. Default is True
    actuators : bool, optional
        Add bad actuator masks and spider. Default is True

    Notes
    -----
    The smallest element in the SCExAO pupil is the bad actuator spider, which is approximately {ACTUATOR_SPIDER_WIDTH*1e3:.1f} mm wide. This is about 0.7\% of the telescope diameter, which means you need to have a miinimum of ~142 pixels across the aperture to sample this element.

    """
    pupil_diameter = PUPIL_DIAMETER * outer
    # make grid over full diameter so undersized pupils look undersized
    max_diam = PUPIL_DIAMETER if outer <= 1 else pupil_diameter

    if pupil_grid is None:
        grid = hp.make_pupil_grid(n, diameter=max_diam)
    else:
        grid = pupil_grid

    # This sets us up with M1+M2, just need to add spiders and DM masks
    pupil_field = hp.make_obstructed_circular_aperture(pupil_diameter, inner)

    # add spiders to field generator
    if spiders:
        spider_width = SPIDER_WIDTH * scale
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))

        # spider in quadrant 1
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET,
                 sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 2
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET,
                 sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 3
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET,
                 -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 4
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET,
                 -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )

    # add actuator masks to field generator
    if actuators:
        # circular masks
        actuator_diameter = ACTUATOR_DIAMETER * scale
        actuator_mask_1 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter,
                                 center=ACTUATOR_OFFSET[0]))
        pupil_field = field_combine(pupil_field, actuator_mask_1)

        actuator_mask_2 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter,
                                 center=ACTUATOR_OFFSET[1]))
        pupil_field = field_combine(pupil_field, actuator_mask_2)

        # spider
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))
        actuator_spider_width = ACTUATOR_SPIDER_WIDTH * scale
        actuator_spider = hp.make_spider(
            ACTUATOR_SPIDER_OFFSET,
            (
                ACTUATOR_SPIDER_OFFSET[0] - cost * pupil_diameter,
                ACTUATOR_SPIDER_OFFSET[1] - sint * pupil_diameter,
            ),
            spider_width=actuator_spider_width,
        )
        pupil_field = field_combine(pupil_field, actuator_spider)

    rotated_pupil_field = hp.make_rotated_aperture(pupil_field,
                                                   np.deg2rad(angle))

    pupil = hp.evaluate_supersampled(rotated_pupil_field, grid, oversample)
    return pupil


def main():
    args = parser.parse_args()
    pupil = generate_pupil(
        args.n,
        outer=args.outer,
        inner=args.inner,
        scale=args.scale,
        angle=args.rotate,
        spiders=args.spiders,
        actuators=args.actuators,
    )
    hdr = fits.Header()
    hdr["PUPDIAM"] = PUPIL_DIAMETER * args.outer, "m, diameter of pupil"
    hdr["OBSTDIAM"] = (
        PUPIL_DIAMETER * args.inner,
        "m, diameter of secondary obstruction",
    )
    if args.spiders:
        hdr["SPIDWDTH"] = SPIDER_WIDTH * args.scale, "m, width of support spiders"
        hdr["SPIDOFF"] = SPIDER_OFFSET, "m, offset of support spiders"
        hdr["SPIDANG"] = args.rotate, "m, angle of spiders with respect to the pupil"
    if args.actuators:
        hdr["ACTDIAM"] = ACTUATOR_DIAMETER * args.scale, "m, diameter of actuator masks"
    fits.writeto(args.filename, pupil, overwrite=True)


if __name__ == "__main__":
    main()