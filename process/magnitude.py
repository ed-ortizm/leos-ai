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
config_file_name = "magnitude.ini"
parser.read(f"{config_file_name}")
# Check files and directory
check = FileDirectory()
# Handle configuration file
# configuration = ConfigurationFile()
###############################################################################
# location of data
data_directory = parser.get("directory", "data")
data_type = parser.get("common", "type")
path_to_files = glob.glob(f"{data_directory}/*/{data_type}/*.fits")

print("Process NaNs and values bellow the median", end="\n")

print("Convert to magnitude scale and mask median pixels", end="\n")

for idx, path_to_file in enumerate(path_to_files):

    file_name = path_to_file.split("/")[-1].split(".")[0]
    print(f"{idx:04d}: {file_name}", end="\r")

    with pyfits.open(path_to_file) as hdu:

        image = hdu[0].data

    # replace NaNs with background
    image = np.where(~np.isfinite(image), np.nanmedian(image), image)
    # replace negative and null counts with median
    image = np.where(image <= 0, np.nanmedian(image), image)
    # compute magnitude
    image = np.log10(image, dtype=np.float32)
    # Set background to zero
    image = np.where(image <= np.median(image), 0., image-np.median(image))
    # Normalize image
    image *= 1/np.max(image)

    save_to = path_to_file.split("/")[:-2]
    save_to = '/'.join(save_to)
    save_to = f"{save_to}/magnitude"
    check.check_directory(save_to, exit_program=False)
    np.save(f"{save_to}/{file_name}.npy", image)

    ###########################################################################
    with open(
        f"{save_to}/{config_file_name}",
        "w", encoding="utf8"
    ) as config_file:

        parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\n Run time: {finish_time-start_time:.2f}", end="\n")
