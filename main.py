import PSpy
import WhiteningFilterspy
import ICApy
import numpy as np
import os

# defining read and write directories


inputFileName = os.path.join(os.getcwd(), "airportSurveillance.hdf5")
resultsDirectory = os.path.join(os.getcwd(), "Results")


# defining images and output file names
averagePSResultsFileName = os.path.join(resultsDirectory, "averagePS.hdf5")
averagePSFigureFileName = os.path.join(resultsDirectory, "averagePS.png")

averagePSLocalResultsFileName = os.path.join(resultsDirectory, "averagePSLocal.hdf5")
averagePSLocalFigureFileName = os.path.join(resultsDirectory, "averagePSLocal.png")

whiteningFiltersFigureFileName = os.path.join(resultsDirectory, "whiteningFilters.png")
whiteningFiltersResultsFileName = os.path.join(resultsDirectory, "whiteningFilters.hdf5")

ICResultsFileName = os.path.join(resultsDirectory, "IC.hdf5")
ICFigureFileName = os.path.join(resultsDirectory, "IC.png")

ICAActivationsResultsFileName = os.path.join(resultsDirectory, "ActivationsICA.hdf5")
ICAActivationsSparsenessFigureFileName = os.path.join(resultsDirectory, "ActivationsICA.png")

# Defining some parameters
sampleSizePS = [32, 32]  # image sample size for the power spectrum
gridSize = [3, 3]
numberOfSamplesPS = 608  # number of samples from the dataset for estimating PS


sampleSizeICA = [12, 12]  # image sample size for the ICA
numberOfSamplesICA = 50000   # number of samples from the dataset for making ICA


## Question 2

averagePS = PSpy.getAveragePS(inputFileName, sampleSizePS, numberOfSamplesPS)

PSpy.saveH5(averagePSResultsFileName, 'averagePS', averagePS)
PSpy.makeAveragePSFigure(averagePS, averagePSFigureFileName)

## Question 3
#averagePSRadial = PSpy.getRadialPS(averagePS)
#radialFreq = PSpy.getRadialFreq(averagePS.shape)
#PSpy.saveH5(averagePSRadialResultsFileName,'averagePSRadial',averagePSRadial)
#PSpy.makeAveragePSRadialFigure(np.unique(radialFreq),averagePSRadial, averagePSRadialFigureFileName)

## Question 4
#averagePSLocal = PSpy.getAveragePSLocal(inputFileName, sampleSize, gridSize)
#PSpy.saveH5(averagePSLocalResultsFileName,'averagePS',averagePSLocal)
#PSpy.makeAveragePSLocalFigure(averagePSLocal, averagePSLocalFigureFileName,gridSize)



## Question 5
#averagePS = PSpy.readH5(averagePSResultsFileName,'averagePS')

#maxPS = np.max(averagePS);
#noiseVarianceList = [maxPS*10**(-9),maxPS*10**(-8),maxPS*10**(-7),maxPS*10**(-6)] #if you do not see anything interesting you can change this values

#whiteningFilters = [];
#for noiseVariance in noiseVarianceList:
#    whiteningFilters.append(WhiteningFilterspy.getPowerSpectrumWhiteningFilter(averagePS,noiseVariance))

#PSpy.saveH5(whiteningFiltersResultsFileName,'whiteningFilters',np.array(whiteningFilters))
#WhiteningFilterspy.makeWhiteningFiltersFigure(whiteningFilters,whiteningFiltersFigureFileName)

## Question 6


#X = ICApy.getICAInputData(inputFileName, sampleSizeICA, numberOfSamplesICA)
#X = ICApy.preprocess(X);
#W = ICApy.getIC(X)

#PSpy.saveH5(ICResultsFileName,'IC',W)
#ICApy.makeIdependentComponentsFigure(W,sampleSizeICA, ICFigureFileName)

## Question 7
#A = ICApy.estimateActivations(W)
#sparsenessMeasure = ICApy.estimateSparseness(A)
#PSpy.saveH5(ICAActivationsResultsFileName,'A',A)
#ICApy.makeSparsenessMeasureFigure(A, ICAActivationsSparsenessFigureFileName)
