import numpy as np
import scipy.fftpack
import os
import h5py
import pylab


def getSampleTopLeftCorner(iMin, iMax, jMin, jMax):
    """ Function that genereates randomly a position between i,j intervals [iMin,iMax], [jMin,jMax]
    Args:
        iMin (int): the i minimum coordinate (i is the column-position of an array)
        iMax (int): the i maximum coordinate (i is the column-position of an array)
        jMin (int): the j minimum coordinate (j is the row-position of an array)
        jMax (int): the j maximum coordinate (j is the row-position of an array)
    Returns:
        [i,j] (tuple(int,int)): random integers such iMin<=i<iMax,jMin<=j<jMax,
    """

    return (np.random.randint(iMin, high=iMax+1), np.random.randint(jMin, high=jMax+1))


def getSampleImage(image, sampleSize, topLeftCorner):
    """ Function that extracts a sample of an image with a given size and a given position
    Args:
        image (numpy.array) : input image to be sampled
        sampleSize (tuple(int,int)): size of the sample
        topLeftCorner (tuple(int,int)): position of the top left corner of the sample within the image
    Returns:
        sample (numpy.array): image sample
    """

    ###write your function here
    # print("You should define the function getSampleImage")
    res = np.zeros(sampleSize)
    for i in range(sampleSize[0]):
        for j in range(sampleSize[1]):
            res[topLeftCorner[0]+i][topLeftCorner[1]+j] = image[topLeftCorner[0]+i][topLeftCorner[1]+j]
    return res


def getSamplePS(sample):
    """ Function that calculates the power spectrum of a image sample
    Args:
        sample (numpy.array): image sample
    Returns:
        samplePS (numpy.array): power spectrum of the sample. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """

    samplePS = np.fft(sample)**2
    np.fft.fftshift(samplePS)
    return samplePS


def getAveragePS(inputFileName, sampleSize, numberOfSamples):
    """ Function that estimates the average power spectrum of a image database
    Args :
        inputFileName (str) : Absolute pathway to the image database stored in the hdf5
        sampleSize (tuple(int,int)): size of the samples that are extrated from the images
        numberOfSamples
    Returns:
        averagePS (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """

    buf = readH5(inputFileName, images)
    res = np.zeros(numberOfSamples)
    for i in range(numberOfSamples):
        topLeftCorner = getSampleTopLeftCorner(0+31*i, 31+31*i, 0+31*i, 31+31*i)
        sample = getSampleImage(buf, sampleSize, topLeftCorner)
        getSamplePS(sample)

    return average


def getRadialFreq(PSSize):
    """ Function that returns the Discrete Fourier Transform radial frequencies
    Args:
        psSize (tuple(int,int)): the size of the window to calculate the frequencies
    Returns:
        radialFreq (numpy.array): radial frequencies in crescent order
    """
    fx = np.fft.fftshift(np.fft.fftfreq(PSSize[0], 1./PSSize[0]))
    fy = np.fft.fftshift(np.fft.fftfreq(PSSize[1], 1./PSSize[1]))
    [X, Y] = np.meshgrid(fx, fy)
    R = np.sqrt(X**2+Y**2)
    return R


def getRadialPS(averagePS):
    """ Function that estimates the average power radial spectrum of a image database
    Args:
        averagePS (numpy.array) : average power spectrum of the database samples.
    Returns:
        averagePSRadial (numpy.array): average radial power spectrum of the database samples.
    """
    print("You should define the function getRadialPS")


def getAveragePSLocal(inputFileName, sampleSize, gridSize):
    """ Function that estimates the local average power spectrum of a image database
    Args:
        inputFileName (str) : Absolute pathway to the image database stored in the hdf5
        sampleSize (tuple(int,int)): size of the samples that are extrated from the images
        gridSize (tuple(int,int)): size of the grid that define the borders of each local region
    Returns:
        averagePSLocal (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see numpy.fft.fftshift)
    """
    ###write your function here
    print("You should define the function getAveragePSLocal")


def makeAveragePSFigure(averagePS, figureFileName):
    """ Function that makes and save the figure with the power spectrum
    Args:
        averagePSLocal (numpy.array): the average power spectrum in an array of shape [sampleShape[0],sampleShape[1]
        figureFileName (str): absolute path where the figure will be saved
    """
    pylab.imshow(np.log(averagePS), cmap="gray")
    pylab.contour(np.log(averagePS))
    pylab.axis("off")
    pylab.savefig(figureFileName)


def makeAveragePSRadialFigure(radialFreq, averagePSRadial, figureFileName):
    """ Function that makes and save the figure with the power spectrum
    Args:
        averagePS (numpy.array) : the average power spectrum
        averagePSRadial (numpy.array): the average radial power spectrum
        figureFileName (str): absolute path where the figure will be saved
    """
    pylab.figure()
    pylab.loglog(radialFreq, averagePSRadial, '.')
    pylab.xlabel("Frequecy")
    pylab.ylabel("Radial Power Spectrum")
    pylab.savefig(figureFileName)


def makeAveragePSLocalFigure(averagePSLocal, figureFileName, gridSize):
    """ Function that makes and save the figure with the local power spectrum
    Args:
        averagePSLocal (numpy.array): the average power spectrum in an array of shape [gridSize[0],gridSize[1],sampleShape[0],sampleShape[1]
        figureFileName (str): absolute path where the figure will be saved
        gridSize (tuple): size of the grid
    """
    pylab.figure()
    for i in range(gridSize[0]):
        for j in range(gridSize[1]):
            pylab.subplot(gridSize[0], gridSize[1], i*gridSize[1]+j+1)
            pylab.imshow(np.log(averagePSLocal[i, j]), cmap="gray")
            pylab.contour(np.log(averagePSLocal[i, j]))
            pylab.axis("off")
    pylab.savefig(figureFileName)


def saveH5(fileName, dataName, numpyArray):
    """ Function that saves numpy arrays in a binary file h5
    Args:
        fileName (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        dataName (str): the dataset name
        numpyArray (numpy.array): the data to be saved
    """

    f = h5py.File(fileName, "w")
    f.create_dataset(dataName, data=numpyArray)
    f.close()


def readH5(fileName, dataName):
    """ Function that reads numpy arrays in a binary file hdf5
    Args:
        fileName (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        dataName (str): the dataset name
    Return:
        numpyArray (numpy.array): the read data
    """
    f = h5py.File(fileName, "r")
    numpyArray = f[dataName][:]
    f.close()
    return numpyArray
