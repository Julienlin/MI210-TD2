import numpy as np
import pylab
from sklearn.decomposition import FastICA
import h5py
import PSpy
from scipy.stats import kurtosis
import matplotlib.pyplot as plt


def getICAInputData(inputFileName, sampleSize, nSamples):
    """ Function that samples the input directory for later to be used by FastICA
    Args:
    inputFileName(str):: Absolute pathway to the image database hdf5 file
    sampleSize (tuple(int,int)): size of the samples that are extrated from the images
    nSamples(int): number of samples that should be taken from the database
    Returns:
    X(numpy.array)nSamples, sampleSize
    """
    # Write your function here
    dataSet = h5py.File(inputFileName, 'r')
    images = dataSet.get('images')
    #print("shape : ",images.shape)
    X = np.zeros((nSamples, sampleSize[0], sampleSize[1]))
    for i in range(nSamples):
        # random image from the dataset
        image_number = np.random.randint(images.shape[0])
        image = images[image_number]
        # random position in the topleftcorner to be sure to have a complete image
        position = PSpy.getSampleTopLeftCorner(
            0, image.shape[0]-sampleSize[0], 0, image.shape[1]-sampleSize[1])
        sample = PSpy.getSampleImage(image, sampleSize, position)
        X[i] = sample
    # print("x",X.shape)
    return X


def preprocess(X):
    """Function that preprocess the data to be fed to the ICA algorithm
    Args:
    X(numpy array): input to be preprocessed
    Returns:
    X(numpy.array)
    """

    if X.ndim > 2:
        X = np.reshape(X, (X.shape[0], X.shape[2]*X.shape[1]))
    for i in range(X.shape[0]):
        X[i] = X[i]-np.mean(X[i])

    return X


def getIC(X):
    """Function that estimates the principal components of the data
    Args:
    X(numpy.array):preprocessed data
    Returns:
    S(numpy.array) the matrix of the independent sources of the data
    """
    ICA = FastICA(algorithm='parallel', whiten=True, max_iter=10000, tol=0.1)
    ICA.fit(X)
    S = ICA.components_
    # S1=ICA.fit_transform(X)
    # print(S.shape)
    return S


def estimateSources(W, X):
    """Function that estimates the independent sources of the data
    Args:
    W(numpy.array):The matrix of the independent components
    X(numpy.array):preprocessed data
    Returns:
    S(numpy.array) the matrix of the sources of X
    """
    S = np.dot(X, W)
    # print(S.shape)
    return S


def estimateSourcesKurtosis(S):
    """Function  that estimates the kurtosis of a set of multivariate random variables
    Args:
    S(numpy array): random variable (n-data points of k-size)
    Returns:
    K (numpy.array): kurtosis of each data point (size n,1)
    """
    kurt = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        kurt[i] = kurtosis(S[i])
    return kurt


def makeKurtosisFigure(S, figureFileName):
    # Write your function here
    print("You should define the functione makeKurtosisFigure")


def makeIdependentComponentsFigure(W, sampleSize, figureFileName):
    W = W.reshape([-1, ]+sampleSize)
    pylab.figure()
    for i in range(W.shape[0]):
        pylab.subplot(sampleSize[0], sampleSize[1], i+1)
        pylab.imshow(W[i], cmap='gray')
        pylab.axis("off")
    pylab.savefig(figureFileName)
