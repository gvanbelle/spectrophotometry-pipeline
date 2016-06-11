__author__ = 'ryan'

from FileSort import *
from os import mkdir
from shutil import copy2
from itertools import chain
from time import time
from numpy import append, zeros, ones, compress, median

colMeds = {}
# creates the trimmed frames directories
trimmedCalibrationFramesDir = directory + 'Trimmed Calibration Frames'
trimmedScienceFramesDir = directory + 'Trimmed Science Frames'
if not exists(trimmedCalibrationFramesDir):
    mkdir(trimmedCalibrationFramesDir)
    mkdir(trimmedScienceFramesDir)
    for file in calibrationFrames:
        fileName = file.replace(directory, '').replace('.fits', '')
        trimmedCalibrationFramesSubDir = trimmedCalibrationFramesDir + '/' + fileName
        mkdir(trimmedCalibrationFramesSubDir)
        copy2(file, trimmedCalibrationFramesSubDir)
    for file in scienceFrames:
        fileName = file.replace(directory, '').replace('.fits', '')
        trimmedScienceFramesSubDir = trimmedScienceFramesDir + '/' + fileName
        mkdir(trimmedScienceFramesSubDir)
        copy2(file, trimmedScienceFramesSubDir)
    trimmedCalibrationFrames = listdir(trimmedCalibrationFramesDir)
    for i, folder in enumerate(trimmedCalibrationFrames):
        trimmedCalibrationFrames[i] = trimmedCalibrationFramesDir + '/' + folder
    trimmedScienceFrames = listdir(trimmedScienceFramesDir)
    for i, folder in enumerate(trimmedScienceFrames):
        trimmedScienceFrames[i] = trimmedScienceFramesDir + '/' + folder
    startTime = time()
    # trims the frames
    for i, folder in enumerate(chain(trimmedCalibrationFrames, trimmedScienceFrames)):
        clock = time()
        print("Trimming, grabbing medians " + folder)
        files = listdir(folder)
        file = folder + '/' + files[0]
        hdulist = fits.open(file, mode='update')
        cropY = append(zeros(42), ones(1968))
        hdulist[0].data = compress(cropY, hdulist[0].data, axis=0)
        cropX = append(zeros(75), ones(1988))
        hdulist[0].data = compress(cropX, hdulist[0].data, axis=1)
        hdulist[0].header['NAXIS1'] = 1988
        hdulist[0].header['NAXIS2'] = 1968
        colMeds[files[0]] = []
        for j in range(hdulist[0].data.shape[1]):
            colMeds[files[0]].append(median(hdulist[0].data[:, j]))
        hdulist.close()
        clock = time() - clock
        totalTime = time() - startTime
        print("Elapsed Time: " + str(round(clock)) + " seconds (Total Time: " + str(round(totalTime)) + " seconds)")
else:
    startTime = time()
    trimmedCalibrationFrames = listdir(trimmedCalibrationFramesDir)
    for i, folder in enumerate(trimmedCalibrationFrames):
        trimmedCalibrationFrames[i] = trimmedCalibrationFramesDir + '/' + folder
    trimmedScienceFrames = listdir(trimmedScienceFramesDir)
    for i, folder in enumerate(trimmedScienceFrames):
        trimmedScienceFrames[i] = trimmedScienceFramesDir + '/' + folder
    for i, folder in enumerate(chain(trimmedCalibrationFrames, trimmedScienceFrames)):
        clock = time()
        print("Grabbing medians " + folder)
        for file in listdir(folder):
            if file.endswith('.fits'):
                fitsFile = file
        # noinspection PyUnboundLocalVariable
        hdulist = fits.open(folder + '/' + fitsFile)
        colMeds[fitsFile] = []
        for j in range(hdulist[0].data.shape[1]):
            colMeds[fitsFile].append(median(hdulist[0].data[:, j]))
        hdulist.close()
        clock = time() - clock
        totalTime = time() - startTime
        print("Elapsed Time: " + str(round(clock)) + " seconds (Total Time: " + str(round(totalTime)) + " seconds)")