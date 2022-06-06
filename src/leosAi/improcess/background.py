"""Utility functions to remove background noise"""

from astropy.stats import sigma_clipped_stats
import numpy as np
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture


def remove_stars(image: np.array)->: np.array:
    """
    Remove stars from background but keeps satellite traces
    if present. For more info check:
        https://photutils.readthedocs.io/en/stable/detection.html

    INPUTS
    image: array with observation image in magnitude units

    OUTPUT
    image_no_stars: array with most of background stars removed
    """

    mean, median, std = sigma_clipped_stats(image, sigma=3.0)

    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)

    sources = daofind(image - median)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

    apertures = CircularAperture(positions, r=4.)
    positions = positions.astype(int)

    image_no_stars = image.copy()

    for pos in positions:
        image_no_stars[pos[1]-3:pos[1]+3, pos[0]-3:pos[0]+3] = 0.

    return image_no_stars
