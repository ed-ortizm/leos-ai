"""Prepare raw images for Fourier Analysis"""

###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import glob
import pickle
import random
import time

import numpy as np
from sklearn.decomposition import IncrementalPCA

from leosAi.utils.managefiles import FileDirectory
###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "ipca.ini"
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

number_of_files = len(path_to_files)
batch_size = parser.getint("pca", "batch_size")
number_of_batches = number_of_files // batch_size

# Complete a batch in case number of batches
# does not fit all files
if number_of_batches % batch_size !=0:

    number_of_batches += 1

    remaining_number_of_files = batch_size - number_of_batches % batch_size
    # randomly pick already used images
    path_to_files += random.choices(path_to_files, k=remaining_number_of_files)



image_shape = np.load(path_to_files[0], mmap_mode="r").shape

batch_shape = (batch_size, ) + image_shape
batch_of_images = np.empty(batch_shape).astype(np.float32)

n_components = parser.getint("pca", "components")
assert n_components <= batch_size
transformer = IncrementalPCA(n_components = n_components)

save_to = f"{data_directory}/gauss_rp"
check.check_directory(save_to, exit_program=False)

for batch in range(number_of_batches):


    # load images to current batch of images
    index_of_images = range(batch_size*batch, batch_size*(batch+1))

    for idx_batch, idx_image in enumerate(index_of_images):

        batch_of_images[idx_batch, ...] = np.load(
            path_to_files[idx_image]
        ).astype(np.float32)

    print(f"IPCA of batch {batch:02d}", end="\n")
    # fit pca
    transformer.fit(batch_of_images.reshape(batch_size, -1))
###############################################################################
with open(f"{save_to}/ipca.pkl", "wb") as file:

    pickle.dump(transformer, file)

###############################################################################
with open(
    f"{save_to}/{config_file_name}",
    "w", encoding="utf8"
) as config_file:

    parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"\n Run time: {finish_time-start_time:.2f}", end="\n")
