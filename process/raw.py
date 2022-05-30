"""Prepare raw images for Fourier Analysis"""

###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import glob
import time

from astropy.io import fits as pyfits
import numpy as np

from leosAi.utils.managefiles import FileDirectory
###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "raw.ini"
parser.read(f"{config_file_name}")
# Check files and directory
check = FileDirectory()
# Handle configuration file
# configuration = ConfigurationFile()
###############################################################################
# location of data
raw_data_directory = parser.get("directory", "raw")
path_to_files = glob.glob(f"{raw_data_directory}/*")

print("Process NaNs and values bellow the median", end="\n")

print("Convert to magnitude scale and mask median pixels", end="\n")


save_data_to = parser.get("directory", "magnitudes")
check.check_directory(save_data_to, exit_program=True)

for idx, path_to_file in enumerate(path_to_files):

    file_name = path_to_file.split("/")[-1].split(".")[0]

    with pyfits.open(path_to_file) as hdu:

        # get magnitude scale to better distinguis objects
        image = np.log10(hdu[0].data)

    # replace NaNs with background
    image = np.where(~np.isfinite(image), np.nanmedian(image), image)
    # Set background to zero
    image = np.where(image <= np.median(image), 0., image-np.median(image))
    # Normalize image
    image *= 1/np.max(image)

    np.save(f"{save_data_to}/{file_name}.npy", image)

###############################################################################
with open(
    f"{save_data_to}/{config_file_name}",
    "w", encoding="utf8"
) as config_file:

    parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
