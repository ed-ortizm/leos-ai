"""Convert difference of images to magnitude and mask over median"""
import os

# disable tensorflow logs: warnings and info :). Allow error logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import time

import numpy as np

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "magnitude.ini"
parser.read(f"{config_file_name}")
# Check files and directory
# check = FileDirectory()
# Handle configuration file
# configuration = ConfigurationFile()
###############################################################################
print("Load images", end="\n")
images_directory = parser.get("directory", "images")
images_file_name = parser.get("file", "images")
images = np.load(f"{images_directory}/{images_file_name}", mmap_mode="r+")

print("Convert to magnitude scale and mask median pixels", end="\n")

for idx, image in enumerate(images):

    print(f"Process image: {idx:03d}", end="\r")

    image = np.abs(image).copy()
    image = np.where(image==0, np.median(image), image)

    image = -2.5 * np.log10(image)
    image = np.where(
        image > np.median(image),
        np.median(image),
        image
    )

    images[idx, ...] = image[...]

save_data_to = parser.get("directory", "magnitudes")
magnitude_name = parser.get("file", "magnitudes")

np.save(f"{save_data_to}/{magnitude_name}", images)
###############################################################################
with open(
    f"{save_data_to}/{config_file_name}",
    "w", encoding="utf8"
) as config_file:

    parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
