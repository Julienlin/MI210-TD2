import numpy as np
import pylab
from sklearn.decomposition import FastICA



def getICAInputData(inputFileName, sampleSize,nSamples):
    """ Function that samples the input directory for later to be used by FastICA
    Args:
    inputFileName(str):: Absolute pathway to the image database hdf5 file
    sampleSize (tuple(int,int)): size of the samples that are extrated from the images
    nSamples(int): number of samples that should be taken from the database
    Returns:
    X(numpy.array)nSamples, sampleSize
    """
    #Write your function here
    print("You should define the function getICAInputData")


def preprocess(X):
    """Function that preprocess the data to be fed to the ICA algorithm
    Args:
    X(numpy array): input to be preprocessed
    Returns:
    X(numpy.array)
    """
    #Write your function here
    print("You should define the function preprocess")


def getIC(X):
    """Function that estimates the principal components of the data
    Args:
    X(numpy.array):preprocessed data
    Returns:
    S(numpy.array) the matrix of the independent sources of the data
    """
    #Write your function here
    print("You should define the function getIC")

def estimateSources(W,X):
    """Function that estimates the independent sources of the data
    Args:
    W(numpy.array):The matrix of the independent components
    X(numpy.array):preprocessed data
    Returns:
    S(numpy.array) the matrix of the sources of X
    """
    #Write your function here
    print("You should define the function estimateSources")


def estimateSourcesKurtosis(S):
    """Function  that estimates the kurtosis of a set of multivariate random variables
    Args: 
    S(numpy array): random variable (n-data points of k-size)
    Returns:
    K (numpy.array): kurtosis of each data point (size n,1) 
    """
    #Write your function here
    print("You should define the functione stimateSourcesKurtosis")

def makeKurtosisFigure(S,figureFileName):
    #Write your function here
    print("You should define the functione makeKurtosisFigure")
        
def makeIdependentComponentsFigure(W, sampleSize,figureFileName): 
    W = W.reshape([-1,]+sampleSize)
    pylab.figure()
    for i in range(W.shape[0]):
        pylab.subplot(sampleSize[0],sampleSize[1],i+1)
        pylab.imshow(W[i],cmap = 'gray')
        pylab.axis("off")
    pylab.savefig(figureFileName)
