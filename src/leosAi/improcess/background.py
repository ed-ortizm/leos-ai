"""Utility functions to remove background noise"""

from astropy.stats import sigma_clipped_stats
import numpy as np
from photutils.detection import DAOStarFinder


def remove_stars(
    image: np.array,
    fwhm: float=3.,
    threshold: int=5,
    radius: int=5)-> np.array:
    """
    Remove stars from background but keeps satellite traces
    if present. For more info check:
        https://photutils.readthedocs.io/en/stable/detection.html

    INPUTS
    image: array with observation image in magnitude units
    fwhm: full width at half maximun for gaussian kernel to use
        in the point spread function of a star
    threshold: number of times the standard deviation of pixels values
        in the image to gauge stars. Pixel values over it are cadidates
        to be stars
    radius: number of pixels from star's centroid to consider
        to set as zero

    OUTPUT
    positions: array with stars' positions in pixels' space
    image_no_stars: array with most of background stars removed
    """

    _, median, std = sigma_clipped_stats(image, sigma=3.0)

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)

    sources = daofind(image - median)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

    positions = positions.astype(int)

    image_no_stars = image.copy()

    for pos in positions:
        image_no_stars[
            pos[1] - radius : pos[1] + radius, pos[0] - radius : pos[0] + 3
        ] = 0.

    return positions, image_no_stars
