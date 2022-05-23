"""Convert difference of images to magnitude and mask over median"""

###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import time

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
print("Load images", end="\n")
images_directory = parser.get("directory", "images")
images_file_name = parser.get("file", "images")
images = np.load(f"{images_directory}/{images_file_name}", mmap_mode="r")

print("Convert to magnitude scale and mask median pixels", end="\n")

save_data_to = parser.get("directory", "magnitudes")

check.check_directory(save_data_to, exit_program=True)

for idx, image in enumerate(images):

    print(f"Process image: {idx:03d}", end="\r")

    image_copy = np.abs(image).copy()

    if np.median(image_copy) == 0.:
        continue

    image_copy = np.where(image_copy==0, np.median(image_copy), image_copy)

    image_copy = -2.5 * np.log10(image_copy)
    image_copy = np.where(
        image_copy > np.median(image_copy),
        np.median(image_copy),
        image_copy
    )

    np.save(f"{save_data_to}/{idx:03d}", image_copy)

###############################################################################
with open(
    f"{save_data_to}/{config_file_name}",
    "w", encoding="utf8"
) as config_file:

    parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
