"""Remove stars from observations that are in magnitude scale"""

###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import glob
import time

import multiprocessing as mp
import numpy as np

from leosAi.improcess.background import remove_stars
from leosAi.utils.managefiles import FileDirectory

def worker(path_to_file):
    """worker"""

    file_name = path_to_file.split("/")[-1].split(".")[0]
    print(file_name)
    image = np.load(path_to_file)
    _, image = remove_stars(image, fwhm=3, threshold=5, radius=6)
    save_to = path_to_file.split("/")[:-2]
    save_to = '/'.join(save_to)
    save_to = f"{save_to}/no_stars"
    check.check_directory(save_to, exit_program=False)
    np.save(f"{save_to}/{file_name}.npy", image)
###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "stars.ini"
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

print("Remove background stars from observations", end="\n")

with mp.Pool(processes=8) as pool:

    pool.map(worker, path_to_files)

# for idx, path_to_file in enumerate(path_to_files):
#
#     file_name = path_to_file.split("/")[-1].split(".")[0]
#     print(f"{idx:04d}: {file_name}", end="\r")
#
#     image = np.load(path_to_file)
#     _, image = remove_stars(image, fwhm=3, threshold=8, radius=5)
#     save_to = path_to_file.split("/")[:-2]
#     save_to = '/'.join(save_to)
#     save_to = f"{save_to}/no_stars"
#     check.check_directory(save_to, exit_program=False)
#     np.save(f"{save_to}/{file_name}.npy", image)
#
#     ###########################################################################
#     with open(
#         f"{save_to}/{config_file_name}",
#         "w", encoding="utf8"
#     ) as config_file:
#
#         parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\n Run time: {finish_time-start_time:.2f}", end="\n")
