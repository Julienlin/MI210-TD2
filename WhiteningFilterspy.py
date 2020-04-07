import os
import numpy as np
import pylab



def truncateNonNeg(x):
    """Function that truncates arrays od real numbers into arrays of non negatives.
    Args:
    x(numpy.array): input array
    Returns:
    y(numpy.array): array with positive or zero numbers
    """
    #Write your function here
    print("You should define the function truncateNonNeg")

def getPowerSpectrumWhiteningFilter(averagePS,noiseVariance):
    """Function that estimates the whitening and denoising power spectrum filter
    Args:
    averagePS(numpy.array): average power spectrum of the observation
    noiseVariance(double): variance of the gaussian white noite.
    Returns:
    w(numpy.array): whitening denoising filter
    """
    #Write your function here
    print("You should define the function getPowerSpectrumWhiteningFilter")


def makeWhiteningFiltersFigure(whiteningFilters,figureFileName):
    pylab.figure()
    for i,whiteningFilter in enumerate(whiteningFilters):
        pylab.subplot(1,len(whiteningFilters),i+1)
        vmax = np.max(np.abs(whiteningFilter))
        vmin = -vmax
        pylab.imshow(whiteningFilter,cmap = 'gray',vmax = vmax, vmin = vmin)
        pylab.axis("off")
    pylab.savefig(figureFileName)

