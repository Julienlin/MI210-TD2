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
    y = np.real(x)
    arraySize = np.shape(y)
    for i in range(arraySize[0]):
        for j in range(arraySize[1]):
            if y[i][j]<0:
                y[i][j]=0.0
    return y

def getPowerSpectrumWhiteningFilter(averagePS,noiseVariance):
    """Function that estimates the whitening and denoising power spectrum filter
    Args:
    averagePS(numpy.array): average power spectrum of the observation
    noiseVariance(double): variance of the gaussian white noite.
    Returns:
    w(numpy.array): whitening denoising filter
    """
    arraySize = np.shape(averagePS)
    temp = np.zeros(arraySize)
    M= arraySize[0]*arraySize[1] #total number of pixels
    for i in range(arraySize[0]):
        for j in range(arraySize[1]):
            temp[i][j] = ((1/np.sqrt(averagePS[i][j]))*((averagePS[i][j]-M*noiseVariance)/averagePS[i][j]))

    w = truncateNonNeg(temp)
    w = np.fft.ifftshift(w)
    w = np.fft.ifft2(w)
    w = truncateNonNeg(w)
    w = np.fft.ifftshift(w)

    return w

def makeWhiteningFiltersFigure(whiteningFilters,figureFileName):
    pylab.figure()
    for i,whiteningFilter in enumerate(whiteningFilters):
        pylab.subplot(1,len(whiteningFilters),i+1)
        vmax = np.max(np.abs(whiteningFilter))
        vmin = -vmax
        pylab.imshow(whiteningFilter,cmap = 'gray',vmax = vmax, vmin = vmin)
        pylab.axis("off")
    pylab.savefig(figureFileName)

