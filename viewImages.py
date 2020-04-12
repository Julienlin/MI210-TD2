import PSpy
import WhiteningFilterspy
import ICApy
import numpy as np
import pylab
import os
import h5py


if __name__ == "__main__":
    # Defining some parameters
    sampleSizePS = [32, 32]  # image sample size for the power spectrum
    gridSize = [3, 3]
    numberOfSamplesPS = 1  # number of samples from the dataset for estimating PS
    inputFileName = os.path.join(os.getcwd(), "airportSurveillance.hdf5")
    resultsDirectory = os.path.join(os.getcwd(), "imagesbis")

    dataset = h5py.File(inputFileName, 'r')
    images = dataset.get('images')

    for i in range(numberOfSamplesPS):
        pylab.imshow(images[i], cmap="gray")
        pylab.axis("off")
        pylab.savefig(os.path.join(resultsDirectory, "image"+str(i)))
