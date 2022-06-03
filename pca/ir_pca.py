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
config_file_name = "ir_pca.ini"
parser.read(f"{config_file_name}")
# Check files and directory
check = FileDirectory()
# Handle configuration file
# configuration = ConfigurationFile()
###############################################################################
# location of data
data_directory = parser.get("directory", "data")
data_type = parser.get("common", "type")
path_to_files = glob.glob(f"{data_directory}/*/{data_type}/*.npy")

for idx, path_to_file in enumerate(path_to_files):

    file_name = path_to_file.split("/")[-1].split(".")[0]
    print(f"{idx:04d}: {file_name}", end="\r")

    image = np.load(path_to_file)
###############################################################################
# with open(
#     f"{save_to}/{config_file_name}",
#     "w", encoding="utf8"
# ) as config_file:
#
#     parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\n Run time: {finish_time-start_time:.2f}", end="\n")
