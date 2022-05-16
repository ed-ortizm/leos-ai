"""Principal Component Analysis on images' observations"""
import numpy as np
from sklearn.decomposition import PCA


class ImagePCA:
    """PCA of imaging data"""

    def __init__(self, number_of_components: int = None):

        if number_of_components is None:

            self.model = PCA()

        else:

            self.model = PCA(number_of_components)
            self.number_of_components = number_of_components

    def train(self, images: np.array) -> None:
        """
            Obtain PCA 'model' of data

            INPUT
            images: batch of images to fit in the model
        """

        self.model.fit(images)

    @property
    def explained_variance(self):
        """
            Return explained variance with the number of componets in the model
        """
        return self.model.explained_variance_ratio_

    def inverse(self, embedding: np.array) -> np.array:
        """
            Obtain image reconstruction of input embedding

            INPUT
            embedding: lower dimensional representation of an
                observation

            OUTPUT

            reconstruction: observation representation of input embedding

        """

        if embedding.ndim == 1:

            embedding = embedding.reshape(1, -1)

        reconstruction = self.model.inverse_transform(embedding)

        return reconstruction

    def predict(self, observations):
        """
            Get embedding of input observations

            INPUT
            observations: batch of telescope observations

            OUTPUTS

            embedding: PCA embedding of observations
        """

        if observations.ndim == 2:

            observations = observations.reshape((1,) + observations.shape)

        embedding = self.model.transform(observations)

        return embedding
