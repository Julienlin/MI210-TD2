import PSpy
import WhiteningFilterspy
import ICApy
import numpy as np
import pylab
import os
import h5py

# Defining some parameters
sampleSizePS = [32, 32]  # image sample size for the power spectrum
gridSize = [3, 3]
numberOfSamplesPS = 608  # number of samples from the dataset for estimating PS
inputFileName = os.path.join(os.getcwd(), "airportSurveillance.hdf5")
resultsDirectory = os.path.join(os.getcwd(), "images")

dataset = h5py.File(inputFileName, 'r')
images = dataset.get('images')

for i in range(608):
    pylab.imshow(images[i], cmap="gray")
    # pylab.contour(image)
    pylab.axis("off")
    pylab.savefig(os.path.join(resultsDirectory, "image"+str(i)))
