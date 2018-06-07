""" Python Utility script to Download the MNIST dataset

    Dataset is used only for Benchmarking

"""
import gzip
import numpy as np
import os
from urllib.request import urlretrieve


class MnistDownloader:

    def __init__(self, directory=None):
        """ Initialisation of the class to download the MNist dataset
            directly from Yann Lecun's website.

            Pass an optional parameter to override the default directory.

        """
        self.URL = 'http://yann.lecun.com/exdb/mnist/'
        self.FILE_NAMES = ['train-images-idx3-ubyte.gz',
                           'train-labels-idx1-ubyte.gz',
                           't10k-images-idx3-ubyte.gz',
                           't10k-labels-idx1-ubyte.gz']

        if directory is None:
            self.DIRECTORY = os.getcwd()

    def setup_directory(self):
        """ Setup the correct directory

            If required, attempt to make all the necessary directories

        """
        if self.DIRECTORY != os.getcwd():

            os.makedirs(self.DIRECTORY, exist_ok=True)

    def download_files(self):
        """ Download all the files that are required by Downloader.

            The script will skip if the zipped file already exists.

        """
        for file in self.FILE_NAMES:
            if file not in os.listdir(self.DIRECTORY):
                urlretrieve(self.URL + file, os.path.join(self.DIRECTORY, file))
                print("Downloaded %s to %s" % (file, self.DIRECTORY))

        self.FILE_NAMES = list(map(lambda x: os.path.join(mnist_csv_downloader.DIRECTORY, x),
                                   mnist_csv_downloader.FILE_NAMES))

    @staticmethod
    def get_image_matrix(path):
        """ Format each image in folder correctly

            Note: The first 16 bytes contain the dimensions of the folder
                  i.e. Number of Images, Row Count, Column Count

            :param path: File path to zipped file
            :return numpy array of the images

        """
        with gzip.open(path) as f:
            pixels = np.frombuffer(f.read(), '>B', offset=16)

        return pixels.reshape(-1, 784).astype('float32') / 255

    @staticmethod
    def get_image_labels(path):
        """ Return the image matrix

            :param path: File path to zipped file
            :return numpy array of the images

        """
        with gzip.open(path) as f:
            integer_labels = np.frombuffer(f.read(), '>B', offset=8)

            # Get the meta data from the file contents
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1

            # Generate the Matrix
            image_matrix = np.zeros((n_rows, n_cols), dtype='bool')
            image_matrix[np.arange(n_rows), integer_labels] = 1

        return image_matrix

    def save_to_csv(self, array, name):
        """ Writing the Arrays to CSV

            :param array:   Numpy array to write into the file
            :param name:    Name of the file to write

        """
        np.savetxt(fname=os.path.join(self.DIRECTORY, name),
                   X=array,
                   delimiter=",",
                   fmt="%10.5f")

    def run(self):
        """ Orchestrate the downloading, extraction and transformation
            of the raw data.
        """
        # Setup and download the files / directories
        self.setup_directory()
        self.download_files()

        # Data preparation
        train_images = self.get_image_matrix(self.FILE_NAMES[0])
        test_images = self.get_image_matrix(self.FILE_NAMES[2])

        train_labels = self.get_image_labels(self.FILE_NAMES[1])
        test_labels = self.get_image_labels(self.FILE_NAMES[3])

        # Write outputs to csv
        self.save_to_csv(np.hstack((train_images, train_labels)), "Train.csv")
        self.save_to_csv(np.hstack((test_images, test_labels)), "yTest.csv")

        return {
            "X_train": train_images, "y_train": train_labels,
            "X_test": test_images, "y_test": test_labels
        }


if __name__ == '__main__':
    print("Running Mnist Downloader")
    mnist_csv_downloader = MnistDownloader()
    output = mnist_csv_downloader.run()
