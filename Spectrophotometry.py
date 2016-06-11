from Astrometry import *
from numpy import amax, asarray, exp, unravel_index, amin, delete
from random import randint
from math import sqrt, floor, ceil, log, isnan, pi, tan
from scipy import integrate
from sys import stdout
import matplotlib.pyplot as plt
from time import gmtime, strftime
from multiprocessing import Process, Lock, Array
from lmfit import Model, Parameters
from operator import add

CPSAM = []
print("Calculating CPS vs AM from Calibration Frames.")
lock = Lock()
wavelengthRefZero = 6500
wavelengthRefStep = 8.78695
spectraMaxWavelength = 10000
spectraMinWavelength = 3000
pixelScale = 0.456
arrInfo = Array('d', 850)  # Used to pass the data between the child[?] processes and the host
superVerbose = 1
plotValuesList = [50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 725, 750]
ccdNonLinear = 58000


def gaussfunc(gaussx, b, c):
    # noinspection PyTypeChecker
    return 1 / c * exp(- log(2) * (gaussx - b) ** 2 / (c ** 2))


def gaussfunc2d(gaussx, gaussy, a, bx, by, c, d):
    val = a * exp(-((gaussx - bx) ** 2 + (gaussy - by) ** 2) / (c ** 2)) + d
    return val.flatten()


def gaussfit2d(box, boxsize):
    a = amax(box)
    bx, by = unravel_index(box.argmax(), box.shape)
    c = 2
    d = amin(box)
    stdout.write('Box size: ' + str(box.shape) + ' boxsize: ' + str(boxsize) + '\n')
    stdout.write('Input [bx,by]: [{:6.1f}'.format(bx) + ', {:6.1f}'.format(by) + ']' + ' Width {:6.1f}'.format(c) +
                 ' Amp {:6.1f}'.format(a) + ' Bkgnd {:6.1f}'.format(d) + '\n')
    gaussxn = []
    gaussyn = []
    for l in range(boxsize):
        for m in range(boxsize):
            gaussxn.append(l)
            gaussyn.append(m)
    gausspopped = 0
    box = box.flatten()
    stdout.write('size of [gaussXn, gaussYn, box]: [{:6.1f}'.format(len(gaussxn)) + ', {:6.1f}'.format(len(gaussyn)) +
                 ', {:6.1f}'.format(box.size) + ']\n')
    for l in range(len(box)):
        if box[l - gausspopped] > ccdNonLinear:
            box = delete(box, l - gausspopped)
            gaussxn.pop(l - gausspopped)
            gaussyn.pop(l - gausspopped)
            gausspopped += 1
    gaussxn = asarray(gaussxn)
    gaussyn = asarray(gaussyn)
    try:
        model = Model(func=gaussfunc2d, independent_vars=["gaussx", "gaussy"], param_names=["a", "bx", "by", "c", "d"])
        params = Parameters()
        params.add('a', value=a, min=0)
        params.add('bx', value=bx, min=0, max=49)
        params.add('by', value=by, min=0, max=49)
        params.add('c', value=c, min=0, max=50)
        params.add('d', value=d, min=d - 50, max=d + 50)
        fit = model.fit(box, gaussx=gaussxn, gaussy=gaussyn, params=params)
        fitvalues = fit.best_values
    except RuntimeError:
        fitreturn = float('nan')
    else:
        stdout.write('[x,y]: [{:6.1f}'.format(fitvalues['bx']) + ', {:6.1f}'.format(fitvalues['by']) + ']' +
                     ' Width {:6.1f}'.format(fitvalues['c']) + ' Amp {:6.1f}'.format(fitvalues['a']) +
                     ' Bkgnd {:6.1f}'.format(fitvalues['d']) + '\n')
        fitreturn = (fitvalues['bx'], fitvalues['by'], fitvalues['c'], fitvalues['a'], fitvalues['d'])
    return fitreturn


def lorentzfunc(lorentzx, b, c):
    return 1 / c / (1 + (lorentzx - b) ** 2 / (c ** 2))
    # return ((c / 2) ** 2 / (lorentzx - b) ** 2 + (c / 2) ** 2)


def pvoigtfunc(pvoigtx, a, b, c, d, e):
    return (e * sqrt(log(2) / pi) * gaussfunc(pvoigtx, b, c) + (1 - e) * 1 / pi * lorentzfunc(pvoigtx, b, c)) * a + d


def pvoigtfunc2(pvoigtx, a, b, c, e):
    return (e * sqrt(log(2) / pi) * gaussfunc(pvoigtx, b, c) + (1 - e) * 1 / pi * lorentzfunc(pvoigtx, b, c)) * a


# noinspection PyTypeChecker
def voigtfit(fitlock, fiti, fitcounts, arrdata, subdir, printtoggle, airmass, fitsfiletext):
    centerspotdenom = 0
    centerspotnum = 0
    for fitj, fittemp in enumerate(fitcounts):
        if fitcounts[fitj] == 0:
            sigma = 1
        else:
            sigma = abs(1 / fitcounts[fitj])
        centerspotnum += fitj / sigma
        centerspotdenom += 1 / sigma
    # starting guess values
    centerspot = centerspotnum / centerspotdenom
    b = int(centerspot + 0.5)                                  # centroid
    c = 4                                           # FWHM
    # d = fitcolmed                            # background level
    # a = (fitcounts[b - 1] + fitcounts[b + 1] + fitcounts[b]) / 3 - d    # amplitude
    a = fitcounts[b]  # - d    # amplitude
    # stdout.write('b: {:3d}'.format(b)+' ')
    e = 0.5
    b = centerspot
    xindex = []
    voigtpopped = 0
    for l in range(80):
        xindex.append(l)
    for l in range(80):
        if fitcounts[l - voigtpopped] > ccdNonLinear:
            fitcounts.pop(l - voigtpopped)
            xindex.pop(l - voigtpopped)
            voigtpopped += 1
    try:
        model = Model(pvoigtfunc2)
        params = Parameters()
        params.add('a', value=a, min=0)
        params.add('b', value=b, min=0, max=80)
        params.add('c', value=c, min=0)
        # params.add('d', value=d, min=d - 25, max=d + 25)
        params.add('e', value=e, min=0, max=1)
        fit = model.fit(asarray(fitcounts), params, pvoigtx=asarray(xindex)).best_values
    except RuntimeError:
        fit = [float('nan'), float('nan'), float('nan'), float('nan')]
        fitreturn = float('nan')
    else:
        fitreturn = integrate.quad(lambda fitx: pvoigtfunc2(fitx, fit['a'], fit['b'], fit['c'], fit['e']),
                                   -20000, 20000)[0]
    fitlock.acquire()
    try:
        if printtoggle:
            stdout.write(fitsfiletext + ' i : {:3d}'.format(fiti) + ' sC: {:8.0f}'.format(fitreturn) +
                         ' AM: {:4.2f}'.format(airmass) + ' ET: {:6.1f}'.format(exposureTime) +
                         ' g0: {:7.1f}'.format(fit['a']) + ' g1: {:5.2f}'.format(fit['b']) +
                         ' g2: {:5.2f}'.format(fit['c']) + ' g4: {:4.2f}'.format(fit['e']) + '\n')
    finally:
        fitlock.release()
    arrdata[fiti] = fitreturn
    if fiti in plotValuesList and superVerbose:
        datax = []
        datay = []
        for l in range(len(xindex)):
            datax.append(xindex[l])
            datay.append(fitcounts[l])
        datax2 = []
        datay2 = []
        for l in range(800):
            datax2.append(l / 10)
            datay2.append(pvoigtfunc2(l / 10, fit['a'], fit['b'], fit['c'], fit['e']))
        datax3 = []
        datay3 = []
        for l in range(80):
            datax3.append(j)
            datay3.append(0)
        # fig = plt.figure()
        plt.figure()
        # ax1 = fig.add_subplot(111)
        plt.scatter(datax[:], datay[:], s=20, c='b', marker="s", label='data')
        plt.scatter(datax2[:], datay2[:], s=2, c='r', marker="o", label='fit')
        plt.scatter(datax3[:], datay3[:], s=1, c='r', marker="o", label='floor')
        plt.text(min(datax[:]), (min(datay[:]) + max(datay[:])) / 2, 'area: {:7.1f}'.format(fitreturn) + '\n' +
                 'AM: {:4.2f}'.format(airmass), style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        plt.legend(loc='upper left')
        plt.savefig(subdir + '/columnPixels-' + str(fiti) + '.png', bbox_inches='tight')
        del datax
        del datay
        plt.clf()


def linefunc(linex, linem, linek):
    return linem * 10 ** (-linek * linex / 2.5)


def linefit(fitam, fitcps, fitweights):
    try:
        model = Model(linefunc)
        params = Parameters()
        params.add('linem', value=5)
        params.add('linek', value=0.25, min=0, max=1)
        fit = model.fit(asarray(fitcps), params, fitweights, linex=asarray(fitam))
        fitvalues = fit.best_values
        fiterr = fit.covar
    except RuntimeError:
        return float('nan'), float('nan')
    return [[fitvalues['linem'], fitvalues['linek']], fiterr]


def wavelength2column(wl, pixelx):
    initx = pixelx - wl / wavelengthRefStep
    intermnlambda = (1 / (wl / 10000)) ** 2
    nlambda = (64.328 + 29498.1 / (146 - intermnlambda) + 255.4 / (41 - intermnlambda)) / 1000000 + 1
    r0 = (nlambda ** 2 - 1) / 2 / nlambda ** 2
    r = (r0 - refR0) * interR
    projr = r * -cos(radians(AZ))
    return initx + projr

# goes through each calibration frame subdirectory
fileNamesShort = []
countsVSFlux = open(directory + 'countsVSFlux.txt', 'w')
spectraPixelRange = []
for subDir in trimmedCalibrationFrames:
    wavelengths = []
    chdir(subDir)
    fitsFile = ''
    # finds the fits file
    for file in listdir(subDir):
        if file.endswith('.fits'):
            fitsFile = file
            fileNamesShort.append(file)
    # grabs the image data
    imageData = fits.open(subDir + '/' + fitsFile)
    image = imageData[0].data
    objectID = imageData[0].header['OBJECT'].strip()
    # opens the Info file
    info = open(subDir + '/Info.txt', 'r')
    # sets the pixel location of the calibrator star to nearest solved (x, y) or the average if unsolved, although
    # calibrators should have a solution
    for line in info.readlines():
        if line.startswith('Zenith Angle'):
            ZA = float(line.split()[-1])
        if line.startswith('Azimuth'):
            AZ = float(line.split()[-1])
        if line.startswith('X'):
            starPixelX = round(float(line.split()[-1])) + 1
        if line.startswith('Y'):
            starPixelY = round(float(line.split()[-1])) + 1
        if line == 'No Solution':
            starPixelX = XAvg + 1
            starPixelY = YAvg + 1
    info.close()
    # noinspection PyUnboundLocalVariable
    oldStarPixelX = starPixelX
    # noinspection PyUnboundLocalVariable
    oldStarPixelY = starPixelY
    starBox = image[starPixelY - 25:starPixelY + 25, starPixelX - 25:starPixelX + 25]
    starBoxSize = starBox.shape[1] - 25
    if starBoxSize != 25:
        starBox = image[starPixelY - starBoxSize:starPixelY + starBoxSize, starPixelX - starBoxSize:starPixelX +
                        starBoxSize]
    fitXY = gaussfit2d(starBox, starBoxSize * 2)
    fitX = fitXY[1] - starBoxSize - 1
    fitY = fitXY[0] - starBoxSize - 1
    starPixelX += fitX
    starPixelY += fitY
    stdout.write('File: [{:s}] \n'.format(fitsFile))
    stdout.write('Old [x,y]: [{:6.1f}'.format(oldStarPixelX) + ', {:6.1f}'.format(oldStarPixelY) + ']' +
                 ' Delta [x,y]: [{:6.1f}'.format(fitX) + ', {:6.1f}'.format(fitY) + ']' +
                 ' New [x,y]: [{:6.1f}'.format(starPixelX) + ', {:6.1f}'.format(starPixelY) + ']' + '\n')
    # start calculations for DCR
    refIntermNLambda = (1 / (wavelengthRefZero / 10000)) ** 2
    refNLambda = (64.328 + 29498.1 / (146 - refIntermNLambda) + 255.4 / (41 - refIntermNLambda)) / 1000000 + 1
    refR0 = (refNLambda ** 2 - 1) / 2 / refNLambda ** 2
    # noinspection PyUnboundLocalVariable
    interR = tan(radians(ZA)) * 206265 / pixelScale
    initWavelength = spectraMinWavelength
    initColumn = wavelength2column(initWavelength, starPixelX)
    firstColumn = ceil(initColumn)
    actualColumn = firstColumn
    prevWavelength = initWavelength - wavelengthRefStep * 2
    prevColumn = wavelength2column(prevWavelength, starPixelX)
    interpWavelength = (prevWavelength - initWavelength) / (prevColumn - initColumn) *\
                       (actualColumn - initColumn) + initWavelength
    wavelengths.append([interpWavelength])
    while initWavelength < spectraMaxWavelength + 50:
        actualColumn -= 1
        initWavelength = interpWavelength + wavelengthRefStep * 2
        initColumn = wavelength2column(initWavelength, starPixelX)
        interpWavelength = (initWavelength - interpWavelength) / (initColumn - (actualColumn + 1)) *\
                           (actualColumn - initColumn) + initWavelength
        wavelengths.append([interpWavelength])
    for i in range(len(wavelengths)):
        if 0 < i < len(wavelengths) - 1:
            currWavelength = wavelengths[i][0]
            shortDiff = currWavelength - wavelengths[i - 1][0]
            wavelengths[i].append(currWavelength - shortDiff / 2)
            longDiff = wavelengths[i + 1][0] - currWavelength
            wavelengths[i].append(currWavelength + longDiff / 2)
            wavelengths[i].append(wavelengths[i][2] - wavelengths[i][1])
    wavelengths.pop(0)
    wavelengths.pop()
    spectraPixelRange.append(len(wavelengths))
    # sets the pixel location of the top right of the spectra box
    spectraPixelX = firstColumn
    spectraPixelY = int(round(starPixelY) + 1) + 40
    spectraPixelYStart = spectraPixelY
    spectralCount = []
    # populates the spectral counts by summing every pixel value in the y for every x for a 800x60 area
    for i in range(spectraPixelRange[-1]):
        spectraPixelY = spectraPixelYStart
        arrInfo[i] = 0
        colMed = colMeds[fitsFile][spectraPixelX]
        for j in range(80):
            if j == 0:
                spectralCount.append([image[spectraPixelY][spectraPixelX] - colMed])
            else:
                spectralCount[i].append(image[spectraPixelY][spectraPixelX] - colMed)
            spectraPixelY -= 1
        spectraPixelX -= 1
        if i in plotValuesList and superVerbose:
            columnPixelsFile = open(subDir + '/columnPixels-' + str(i) + '.txt', 'w')
            for j in range(80):
                columnPixelsFile.write('j:\t' + str(j) + '\tValue:\t' + str(spectralCount[i][j]) + '\n')
            columnPixelsFile.close()
    stdSpectralCount = []
    wavelength = spectraMinWavelength
    for i, count in enumerate(spectralCount):
        if i == 0:
            longPortion = wavelengths[i][2] - wavelength
            longPercent = longPortion / wavelengths[i][3]
            stdSpectralCount.append([j * longPercent for j in count])
            wavelength += 10
        elif wavelengths[i][2] >= spectraMaxWavelength:
            shortPortion = wavelength - wavelengths[i][1]
            shortPercent = shortPortion / wavelengths[i][3]
            stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1], [j * shortPercent for j in count]))
            break
        else:
            if wavelength > wavelengths[i][2]:
                stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1], count))
            else:
                shortPortion = wavelength - wavelengths[i][1]
                shortPercent = shortPortion / wavelengths[i][3]
                stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1], [j * shortPercent for j in count]))
                stdSpectralCount.append([j * (1 - shortPercent) for j in count])
                wavelength += 10
    airMass = imageData[0].header['AIRMASS']
    exposureTime = imageData[0].header['EXPTIME']
    solvingProcs = []
    for i in range(len(stdSpectralCount)):
        # voigtfit(lock, i, spectralCount[i][0], arrInfo, 1)
        # skips multiprocessing for debug purposes
        if len(solvingProcs) < cpuCount:
            p = Process(target=voigtfit, args=(lock, i, stdSpectralCount[i], arrInfo, subDir, 1, airMass, fitsFile))
            p.start()
            solvingProcs.append(p)
            while len(solvingProcs) == cpuCount:
                for proc in solvingProcs:
                    if not proc.is_alive():
                        solvingProcs.remove(proc)
                        break
    for proc in solvingProcs:
        while proc.is_alive():
            continue
    # noinspection PyUnboundLocalVariable
    # p.join()        # tab out to the 'for' multiprocessing
    stdSpectralCountErr = []
    for i in range(len(stdSpectralCount)):
        temp = arrInfo[i]
        if temp > 0:
            stdSpectralCount[i] = temp
            stdSpectralCountErr.append(sqrt(temp))
        else:
            stdSpectralCount[i] = float('nan')
            stdSpectralCountErr.append(float('nan'))
    countsPerSecond = []
    countsPerSecondErr = []
    # calculates cps from the spectral counts and the exposure time
    for i in range(len(stdSpectralCount)):
        countsPerSecond.append(stdSpectralCount[i] / exposureTime)
        countsPerSecondErr.append(stdSpectralCountErr[i] / exposureTime)
    imageData.close()
    countsVSFlux.write('Calibrator:\t' + objectID + '\n')
    for i, counts in enumerate(countsPerSecond):
        countsVSFlux.write('i:\t' + str(i) + '\tCPS:\t' + str(counts) + '\n')
    CPSAM.append([countsPerSecond, countsPerSecondErr, airMass, objectID])
stdout.write('\n')
# grabs the true flux vs wavelength of the calibrator from the h_stis_ngsl files
if exists('/home/ryan/PycharmProjects/Spectrophotometric Data Reduction Pipeline/'):
    chdir('/home/ryan/PycharmProjects/Spectrophotometric Data Reduction Pipeline/')
else:
    chdir('/home/pti/data-SP/stis_ngsl_v2/')
realFluxAvgsCal = []
bestFitLine = []
for calibrator in calibrationObjects:
    calibratorFitsFile = 'h_stis_ngsl_' + calibrator.lower() + '_v2.fits'
    if exists(calibratorFitsFile):
        calibratorData = fits.open(calibratorFitsFile)
    else:
        calibratorData = fits.open('h_stis_ngsl_' + calibrator.replace('+', '').lower() + '_v2.fits')
    calFluxVsWavelength = calibratorData[1].data
    calibratorData.close()
    # sets the wavelengths for use, including min, max, and initializing the current
    wavelengthIndex = 0
    wavelength = spectraMinWavelength
    while calFluxVsWavelength[wavelengthIndex][0] < wavelength:
        wavelengthIndex += 1
    maxWavelength = calFluxVsWavelength[-1][0]  # max wavelength of NGSL, typically 10196A
    minWavelength = calFluxVsWavelength[0][0]  # min wavelength of NGSL, typically 1676A
    stdout.write('Min/Max wavelengths : ' + str(minWavelength) + ' / ' + str(maxWavelength) + '\n')
    # goes through each x in the spectra box
    realFluxAvgs = []
    # only processes the x values which give a wavelength value that is present in the true data
    while minWavelength <= wavelength <= maxWavelength:
        # calculates partial wavelengths and fluxes because the pixels do not line up perfectly with the true
        # wavelength values
        actualWavelength = calFluxVsWavelength[wavelengthIndex][0]
        lowWavelength = calFluxVsWavelength[wavelengthIndex - 1][0]
        partialLowWavelength = (actualWavelength - wavelength) / (actualWavelength - lowWavelength)
        realFlux = calFluxVsWavelength[wavelengthIndex - 1][1] * partialLowWavelength
        realFluxErr = (calFluxVsWavelength[wavelengthIndex - 1][2] ** 2) * (partialLowWavelength ** 2)
        realFluxCount = partialLowWavelength
        wavelength += 10
        while wavelengthIndex < len(calFluxVsWavelength) - 1 and calFluxVsWavelength[wavelengthIndex][0] <\
                wavelength:
            realFlux += calFluxVsWavelength[wavelengthIndex][1]
            realFluxErr += calFluxVsWavelength[wavelengthIndex][2] ** 2
            realFluxCount += 1
            wavelengthIndex += 1
        highWavelength = calFluxVsWavelength[wavelengthIndex][0]
        actualWavelength = calFluxVsWavelength[wavelengthIndex - 1][0]
        partialHighWavelength = (wavelength - actualWavelength) / (highWavelength - actualWavelength)
        realFlux += calFluxVsWavelength[wavelengthIndex][1] * partialHighWavelength
        realFluxErr += (calFluxVsWavelength[wavelengthIndex][2] ** 2) * (partialHighWavelength ** 2)
        realFluxCount += partialHighWavelength
        realFluxAvg = realFlux / realFluxCount                  # wtd avg of ergs/cm^2/s/A
        realFluxAvgErr = sqrt(realFluxErr) / realFluxCount
        realFluxAvg *= 10                        # ergs/cm^2/s/A -> ergs/cm^2/s
        realFluxAvgErr *= 10
        realFluxAvgs.append((realFluxAvg, realFluxAvgErr))
    countsVSFlux.write('Calibrator:\t' + calibrator + '\n')
    for i, flux in enumerate(realFluxAvgs):
        countsVSFlux.write('i:\t{:3d}'.format(i) +
                           '\tWvlen:\t{:6.1f}'.format(wavelength) +
                           '\tFlux:\t{:7.2e}'.format(flux[0]) + '\n')
    realFluxAvgsCal.append((realFluxAvgs, calibrator))
countsVSFlux.close()
for i in range(len(stdSpectralCount)):
    CPS = []
    CPSErr = []
    AM = []
    lineFitWeights = []
    # grabs the cps and am at that x
    stdout.write('i : {:0>3d}'.format(i) + ' ')
    indexCPS = 0
    breakLoop = False
    for j in CPSAM:
        temp = j[0][i]
        fluxAvg = []
        # for calibrator in logRealFluxAvgsCal:
        for calibrator in realFluxAvgsCal:
            if j[3] == calibrator[1]:
                try:
                    fluxAvg = calibrator[0][i]
                except IndexError:
                    breakLoop = True
                break
        if breakLoop:
            break
        temp2 = temp / fluxAvg[0] / 1e12    # units of counts/second / flux and re-normalize the numbers (temporarily)
        stdout.write('FPCPS AM: {:6.4f}'.format(temp2) + ' {:6.4f}'.format(j[2]) + ' ')
        if not isnan(temp2):
            CPS.append(temp2)
            #
            # see error propagation formula for division:
            # http://www2.lowell.edu/users/gerard/q_refs/error_prop.html
            #
            CPSErr.append(sqrt((j[1][i] ** 2 / temp ** 2 + fluxAvg[1] ** 2 / fluxAvg[0] ** 2) * temp2 ** 2))
            lineFitWeights.append(1 / (CPSErr[indexCPS] ** 2))
            indexCPS += 1
            AM.append(j[2])
    if breakLoop:
        break
    # finds the best fit line
    try:
        bestFitLine.append(linefit(AM, CPS, lineFitWeights))
    except TypeError:
        stdout.write('fit : nan nan \n')
        bestFitLine.append(([float('nan'), float('nan')],
                            [[float('nan'), float('nan')], [float('nan'), float('nan')]]))
    else:
        stdout.write('fit : {:6.4f}'.format(bestFitLine[i][0][0]))
        try:
            stdout.write(' +- {:6.4f}'.format(sqrt(abs(bestFitLine[i][1][0][0]))))
        except TypeError:
            bestFitLine[i][1] = [[float('nan'), float('nan')], [float('nan'), float('nan')]]
        stdout.write(' {:6.4f}'.format(bestFitLine[i][0][1]))
        stdout.write(' +- {:6.4f}'.format(sqrt(abs(bestFitLine[i][1][1][1]))))
        stdout.write('\n')
    if i in plotValuesList:
        dataX = []
        dataY = []
        dataError = []
        dataIndex = 0
        for j in CPS:
            temp = bestFitLine[i][0][1]
            dataX.append(10 ** (temp * AM[dataIndex] / -2.5))
            dataY.append(CPS[dataIndex])
            dataError.append(CPSErr[dataIndex])
            dataIndex += 1
        plt.subplots_adjust(bottom=0.1)
        # plt.scatter(data[:, 0], data[:, 1], marker = 'o')
        plt.errorbar(dataX[:], dataY[:], yerr=dataError, fmt='o')
        plt.text(min(dataX[:]), min(dataY[:]), 'error bars as-is', style='italic',
                 bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        for label, x, y in zip(fileNamesShort, dataX[:], dataY[:]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(50 + randint(0, 1) * -100, 20 + randint(-50, 50)),
                textcoords='offset points', ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        # plt.show()
        plt.savefig(directory + 'CPSAM-' + str(i) + '.png', bbox_inches='tight')
        del dataX
        del dataY
        del dataError
        plt.clf()
slopes = []
slopeErrs = []
intercepts = []
interceptErrs = []
for line in bestFitLine:
    # if line[0][0] > sqrt(abs(line[1][0][0])):
    slopes.append(line[0][0])
    slopeErrs.append(sqrt(abs(line[1][0][0])))
    # if line[0][1] > sqrt(abs(line[1][1][1])):
    intercepts.append(line[0][1])
    interceptErrs.append(sqrt(abs(line[1][1][1])))
plt.figure()
#plt.plot(range(len(slopes)), slopes)
#plt.scatter(range(len(slopes)), slopes, marker = 'o')
plt.errorbar(range(len(slopes)), slopes, yerr=slopeErrs, fmt='o')
#plt.ylim(0,0.4)
plt.savefig(directory + 'slopes.png', bbox_inches='tight')
plt.figure()
#plt.plot(range(len(intercepts)), intercepts)
#plt.scatter(range(len(intercepts)), intercepts, marker = 'o')
plt.errorbar(range(len(intercepts)), intercepts, yerr=interceptErrs, fmt='o')
plt.ylim(0, 1.5)
plt.savefig(directory + 'k-values.png', bbox_inches='tight')
# goes through each science frame directory
for subDir in trimmedScienceFrames:
    clock = time()
    print("Applying Counts to Real Flux Ratio to " + subDir)
    chdir(subDir)
    fitsFile = ''
    # finds the fits file
    for file in listdir(subDir):
        if file.endswith('.fits'):
            fitsFile = file
    # grabs the image data
    imageData = fits.open(subDir + '/' + fitsFile)
    image = imageData[0].data
    objectID = imageData[0].header['OBJECT'].strip()
    # opens the Info file
    info = open(subDir + '/Info.txt', 'r')
    targetPixelX = 0
    targetPixelY = 0
    solved = True
    # sets the pixel location of the science star to nearest solved (x, y) or the average if unsolved
    for line in info.readlines():
        if line.startswith('Zenith Angle'):
            ZA = float(line.split()[-1])
        if line.startswith('Azimuth'):
            AZ = float(line.split()[-1])
        if line.startswith('X'):
            targetPixelX = round(float(line.split(' ')[-1][:-2]))
        if line.startswith('Y'):
            targetPixelY = round(float(line.split(' ')[-1][:-2]))
        if line == 'No Solution':
            targetPixelX = XAvg + 1
            targetPixelY = YAvg + 1
            solved = False
    info.close()
    starBox = image[targetPixelY - 25:targetPixelY + 25, targetPixelX - 25:targetPixelX + 25]
    starBoxSize = starBox.shape[1] - 25
    if starBoxSize != 25:
        starBox = image[targetPixelY - starBoxSize:targetPixelY + starBoxSize,
                        targetPixelX - starBoxSize:targetPixelX + starBoxSize]
    fitXY = gaussfit2d(starBox, starBoxSize * 2)
    fitX = fitXY[1] - starBoxSize - 1
    fitY = fitXY[0] - starBoxSize - 1
    targetPixelX += fitX
    targetPixelY += fitY
    # gets the exposure time from the image header
    exposureTime = imageData[0].header['EXPTIME']
    # gets the air mass from the image header
    airMass = imageData[0].header['AIRMASS']
    # does this if solved
    if solved:
        xylsFile = ''
        # grabs the xyls file
        for file in listdir(subDir):
            if file.endswith('.xyls'):
                xylsFile = file
        xylsFile = subDir + '/' + xylsFile
        starXY = fits.open(xylsFile)
        # grabs the list of stars from the xyls file
        stars = starXY[1].data
        # goes through each star
        for i, xy in enumerate(stars):
            wavelengths = []
            # only processes stars whose spectra are actually contained in the image
            if 530 <= xy[0] < 1989 and 41 <= xy[1] < 1927:
                stdout.write('Input [bx,by]: [{:6.1f}'.format(xy[0]) + ', {:6.1f}'.format(xy[1]) + ']' + '\n')
                starBox = image[xy[1] - 25:xy[1] + 25, xy[0] - 25:xy[0] + 25]
                starBoxSize = starBox.shape[1] - 25             # // is integer division
                if starBoxSize != 25:
                    starBox = image[xy[1] - starBoxSize:xy[1] + starBoxSize, xy[0] - starBoxSize:xy[0] + starBoxSize]
                fitXY = gaussfit2d(starBox, starBoxSize * 2)
                fitX = fitXY[1] - starBoxSize - 1
                fitY = fitXY[0] - starBoxSize - 1
                xy[0] += fitX
                xy[1] += fitY
                # start calculations for DCR
                refIntermNLambda = (1 / (wavelengthRefZero / 10000)) ** 2
                refNLambda = (64.328 + 29498.1 / (146 - refIntermNLambda) + 255.4 / (41 - refIntermNLambda)) / 1000000\
                    + 1
                refR0 = (refNLambda ** 2 - 1) / 2 / refNLambda ** 2
                # noinspection PyUnboundLocalVariable
                interR = tan(radians(ZA)) * 206265 / pixelScale
                initWavelength = spectraMinWavelength
                initColumn = wavelength2column(initWavelength, xy[0])
                firstColumn = ceil(initColumn)
                actualColumn = firstColumn + 1
                prevWavelength = initWavelength - wavelengthRefStep * 2
                prevColumn = wavelength2column(prevWavelength, xy[0])
                interpWavelength = (prevWavelength - initWavelength) / (prevColumn - initColumn) *\
                                   (actualColumn - initColumn) + initWavelength
                wavelengths.append([interpWavelength])
                while initWavelength < spectraMaxWavelength + 50:
                    actualColumn -= 1
                    initWavelength = interpWavelength + wavelengthRefStep * 2
                    initColumn = wavelength2column(initWavelength, xy[0])
                    interpWavelength = (initWavelength - interpWavelength) / (initColumn - (actualColumn + 1)) *\
                                       (actualColumn - initColumn) + initWavelength
                    wavelengths.append([interpWavelength])
                for j in range(len(wavelengths)):
                    if 0 < j < len(wavelengths) - 1:
                        currWavelength = wavelengths[j][0]
                        shortDiff = currWavelength - wavelengths[j - 1][0]
                        wavelengths[j].append(currWavelength - shortDiff / 2)
                        longDiff = wavelengths[j + 1][0] - currWavelength
                        wavelengths[j].append(currWavelength + longDiff / 2)
                        wavelengths[j].append(wavelengths[j][2] - wavelengths[j][1])
                wavelengths.pop(0)
                wavelengths.pop()
                spectraPixelRange = len(wavelengths)
                # sets the pixel location of the top right of the spectra box
                spectraPixelX = firstColumn
                spectraPixelY = xy[1] + 40
                spectraPixelYStart = spectraPixelY
                spectralCount = []
                # populates the spectral counts by summing every pixel value in the y for every x for a 800x60 area
                for j in range(spectraPixelRange):
                    spectraPixelY = spectraPixelYStart
                    arrInfo[i] = 0
                    colMed = colMeds[fitsFile][spectraPixelX]
                    if spectraPixelX > 0:
                        for k in range(80):
                            if k == 0:
                                spectralCount.append([image[spectraPixelY][spectraPixelX] - colMed])
                            else:
                                spectralCount[j].append(image[spectraPixelY][spectraPixelX] - colMed)
                            spectraPixelY -= 1
                    spectraPixelX -= 1
                    if not spectraPixelX:
                        break
                stdSpectralCount = []
                wavelength = spectraMinWavelength
                for j, count in enumerate(spectralCount):
                    if j == 0:
                        longPortion = wavelengths[j][2] - wavelength
                        longPercent = longPortion / wavelengths[j][3]
                        stdSpectralCount.append([k * longPercent for k in count])
                        wavelength += 10
                    elif wavelengths[j][2] >= spectraMaxWavelength:
                        shortPortion = wavelength - wavelengths[j][1]
                        shortPercent = shortPortion / wavelengths[j][3]
                        stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1], [k * shortPercent for k in count]))
                        break
                    else:
                        if wavelength > wavelengths[j][2]:
                            stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1], count))
                        else:
                            shortPortion = wavelength - wavelengths[j][1]
                            shortPercent = shortPortion / wavelengths[j][3]
                            stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1],
                                                            [k * shortPercent for k in count]))
                            stdSpectralCount.append([k * (1 - shortPercent) for k in count])
                            wavelength += 10
                solvingProcs = []
                for j in range(len(stdSpectralCount)):
                    if len(solvingProcs) < cpuCount:
                        p = Process(target=voigtfit, args=(lock, j, stdSpectralCount[j], arrInfo, subDir, 1, airMass,
                                                           fitsFile))
                        p.start()
                        solvingProcs.append(p)
                        while len(solvingProcs) == cpuCount:
                            for proc in solvingProcs:
                                if not proc.is_alive():
                                    solvingProcs.remove(proc)
                                    break
                for proc in solvingProcs:
                    while proc.is_alive():
                        continue
                stdSpectralCountErr = []
                for j in range(len(stdSpectralCount)):
                    temp = arrInfo[j]
                    if temp > 0:
                        stdSpectralCount[j] = temp
                        stdSpectralCountErr.append(sqrt(temp))
                    else:
                        stdSpectralCount[j] = float('nan')
                        stdSpectralCountErr.append(float('nan'))
                countsPerSecond = []
                countsPerSecondErr = []
                for j in range(len(stdSpectralCount)):
                    countsPerSecond.append(stdSpectralCount[j] / exposureTime)
                    countsPerSecondErr.append(stdSpectralCountErr[j] / exposureTime)
                fileName = ''
                # sets the file name to target if (x, y) of star matches (x, y) of target
                if round(xy[0]) == round(targetPixelX) and round(xy[1]) == round(targetPixelY):
                    fileName = '/' + objectID + '.phot'
                    photFileID = objectID
                # sets the file name to stars RA DE if not target
                else:
                    rdlsFile = ''
                    # grabs the rdls file
                    for file in listdir(subDir):
                        if file.endswith('.rdls'):
                            rdlsFile = file
                    rdlsFile = subDir + '/' + rdlsFile
                    # grabs the solved RA DE for each star
                    starRADE = fits.open(rdlsFile)
                    RADE = starRADE[1].data[i]
                    normalRA = RADE[0]
                    normalDE = RADE[1]
                    # converts those values to normal forms
                    normalRAHour, normalRAMin = divmod(normalRA, 15)
                    normalRAMin *= 4
                    normalRASec = normalRAMin - floor(normalRAMin)
                    normalRAMin = floor(normalRAMin)
                    normalRASec *= 60
                    if normalDE >= 0:
                        normalDEDeg = floor(normalDE)
                    else:
                        normalDEDeg = ceil(normalDE)
                    normalDEArcMin = abs(normalDE) - abs(normalDEDeg)
                    normalDEArcMin *= 60
                    normalDEArcSec = normalDEArcMin - floor(normalDEArcMin)
                    normalDEArcMin = floor(normalDEArcMin)
                    normalDEArcSec *= 60
                    normalRAHour = int(normalRAHour)
                    normalRAMin = int(normalRAMin)
                    normalDEDeg = int(normalDEDeg)
                    normalDEArcMin = int(normalDEArcMin)
                    normalRADE = [normalRAHour, normalRAMin, normalRASec, normalDEDeg, normalDEArcMin, normalDEArcSec]
                    fileName = '/{:02d}_{:02d}_{:05.2f}_{:+02d}_{:02d}_{:05.2f}.phot'.format(*normalRADE)
                    photFileID = '{:02d}{:02d}{:05.2f}{:+02d}{:02d}{:05.2f}'.format(*normalRADE)
                # creates the wavelength and flux file
                starInfo = open(subDir + fileName, 'w')
                starInfo.write('# objectID\tWavelength\tDelta Wavelength\tFlux\tFluxErr')
                # initializes the wavelength to the far right of the spectra box
                wavelength = wavelengths[0][0]
                ratioIndex = 0
                # goes through each count and x value pair
                # for j in range(len(logCountsPerSecond)):
                plotWvlen = []
                plotFlux = []
                plotFluxErr = []
                for j in range(len(countsPerSecond)):
                    # only processes wavelengths which are in the true data
                    if minWavelength <= wavelength <= maxWavelength:
                        starInfo.write('\n')
                        zeroAMcountCal = bestFitLine[ratioIndex][0][0]  # units of counts/second / flux / 1e12
                        zeroAMcountCalErr = sqrt(abs(bestFitLine[ratioIndex][1][0][0]))  # units of counts/second / flux
                        zeroAMcountCal *= 1e12                                           # / 1e12
                        zeroAMcountCalErr *= 1e12
                        zeroAMcount = countsPerSecond[j] / zeroAMcountCal
                        try:
                            zeroAMcountErr = sqrt((countsPerSecondErr[j] ** 2 / countsPerSecond[j] ** 2 +
                                                   zeroAMcountCalErr ** 2 / zeroAMcountCal ** 2)) * zeroAMcount
                        except OverflowError:
                            zeroAMcount = float('nan')
                            zeroAMcountErr = float('nan')
                        flux = zeroAMcount
                        fluxErr = zeroAMcountErr
                        flux /= wavelengths[j][0]  # convert back from total flux to flux per A
                        fluxErr /= wavelengths[j][0]
                        flux *= 10000  # convert from per A to per um
                        fluxErr *= 10000
                        ratioIndex += 1
                        if isnan(flux):
                            starInfo.write('# ')
                        elif fluxErr / flux > 0.5:
                            starInfo.write('# ')
                        else:
                            plotWvlen.append(wavelength / 10)
                            plotFlux.append(flux)
                            plotFluxErr.append(fluxErr)
                        starInfo.write('Fl ' + photFileID + ' {:8.3f}'.format(wavelength / 10) +
                                       ' {:5.4f}'.format(wavelengths[j][3] / 10) + ' {:6.4e}'.format(flux) +
                                       ' {:6.4e}'.format(fluxErr) + ' # test data from 31-inch pipelineSP, run ' +
                                       strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
                    wavelength = wavelengths[j + 1][0]
                    # increments wavelength either way
                starInfo.write('\n')
                starInfo.close()
                plotFluxMedian = median(plotFlux)
                if (len(plotWvlen)) > 0:
                    popped = 0
                    for j in range(len(plotFlux)):
                        # noinspection PyTypeChecker
                        if plotFlux[j - popped] > 5 * plotFluxMedian:
                            plotWvlen.pop(j - popped)
                            plotFlux.pop(j - popped)
                            plotFluxErr.pop(j - popped)
                            popped += 1
                    plt.figure()
                    #plt.plot(range(len(slopes)), slopes)
                    #plt.scatter(range(len(slopes)), slopes, marker = 'o')
                    plt.errorbar(plotWvlen, plotFlux, yerr=plotFluxErr, fmt='o')
                    #plt.ylim(0,0.4)
                    plt.savefig(subDir + fileName + '.png', bbox_inches='tight')
                    plt.clf()
    # does this if unsolved
    else:
        wavelengths = []
        # sets the pixel location of the top right of the spectra box
        info = open(subDir + '/Info.txt', 'r')
        for line in info.readlines():
            if line.startswith('StarID'):
                objectID = line.split(' ')[-1][:-1]
        info.close()
        # start calculations for DCR
        refIntermNLambda = (1 / (wavelengthRefZero / 10000)) ** 2
        refNLambda = (64.328 + 29498.1 / (146 - refIntermNLambda) + 255.4 / (41 - refIntermNLambda)) / 1000000 + 1
        refR0 = (refNLambda ** 2 - 1) / 2 / refNLambda ** 2
        # noinspection PyUnboundLocalVariable
        interR = tan(radians(ZA)) * 206265 / pixelScale
        initWavelength = spectraMinWavelength
        initColumn = wavelength2column(initWavelength, targetPixelX)
        firstColumn = ceil(initColumn)
        actualColumn = firstColumn + 1
        prevWavelength = initWavelength - wavelengthRefStep * 2
        prevColumn = wavelength2column(prevWavelength, targetPixelX)
        interpWavelength = (prevWavelength - initWavelength) / (prevColumn - initColumn) *\
                           (actualColumn - initColumn) + initWavelength
        wavelengths.append([interpWavelength])
        while initWavelength < spectraMaxWavelength + 50:
            actualColumn -= 1
            initWavelength = interpWavelength + wavelengthRefStep * 2
            initColumn = wavelength2column(initWavelength, targetPixelX)
            interpWavelength = (initWavelength - interpWavelength) / (initColumn - (actualColumn + 1)) *\
                               (actualColumn - initColumn) + initWavelength
            wavelengths.append([interpWavelength])
        for i in range(len(wavelengths)):
            if 0 < i < len(wavelengths) - 1:
                currWavelength = wavelengths[i][0]
                shortDiff = currWavelength - wavelengths[i - 1][0]
                wavelengths[i].append(currWavelength - shortDiff / 2)
                longDiff = wavelengths[i + 1][0] - currWavelength
                wavelengths[i].append(currWavelength + longDiff / 2)
                wavelengths[i].append(wavelengths[i][2] - wavelengths[i][1])
        wavelengths.pop(0)
        wavelengths.pop()
        spectraPixelRange = len(wavelengths)
        # sets the pixel location of the top right of the spectra box
        spectraPixelX = firstColumn
        spectraPixelY = targetPixelY + 40
        spectraPixelYStart = spectraPixelY
        spectralCount = []
        # populates the spectral counts by summing every pixel value in the y for every x for a 800x60 area
        for i in range(spectraPixelRange):
            spectraPixelY = spectraPixelYStart
            arrInfo[i] = 0
            colMed = colMeds[fitsFile][spectraPixelX]
            for j in range(80):
                if j == 0:
                    spectralCount.append([image[spectraPixelY][spectraPixelX] - colMed])
                else:
                    spectralCount[i].append(image[spectraPixelY][spectraPixelX] - colMed)
                spectraPixelY -= 1
            spectraPixelX -= 1
            if not spectraPixelX:
                break
        stdSpectralCount = []
        wavelength = spectraMinWavelength
        for i, count in enumerate(spectralCount):
            if i == 0:
                longPortion = wavelengths[i][2] - wavelength
                longPercent = longPortion / wavelengths[i][3]
                stdSpectralCount.append([j * longPercent for j in count])
                wavelength += 10
            elif wavelengths[i][2] >= spectraMaxWavelength:
                shortPortion = wavelength - wavelengths[i][1]
                shortPercent = shortPortion / wavelengths[i][3]
                stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1], [j * shortPercent for j in count]))
                break
            else:
                if wavelength > wavelengths[i][2]:
                    stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1], count))
                else:
                    shortPortion = wavelength - wavelengths[i][1]
                    shortPercent = shortPortion / wavelengths[i][3]
                    stdSpectralCount[-1] = list(map(add, stdSpectralCount[-1], [j * shortPercent for j in count]))
                    stdSpectralCount.append([j * (1 - shortPercent) for j in count])
                    wavelength += 10
        superVerbose = 0
        solvingProcs = []
        for i in range(len(stdSpectralCount)):
            if len(solvingProcs) < cpuCount:
                p = Process(target=voigtfit, args=(lock, i, stdSpectralCount[i], arrInfo, subDir, 1, airMass, fitsFile))
                p.start()
                solvingProcs.append(p)
                while len(solvingProcs) == cpuCount:
                    for proc in solvingProcs:
                        if not proc.is_alive():
                            solvingProcs.remove(proc)
                            break
        for proc in solvingProcs:
            while proc.is_alive():
                continue
        spectralCountErr = []
        for i in range(len(stdSpectralCount)):
            temp = arrInfo[i]
            if temp > 0:
                stdSpectralCount[i] = temp
                stdSpectralCountErr.append(sqrt(temp))
            else:
                stdSpectralCount[i] = float('nan')
                stdSpectralCountErr.append(float('nan'))
        countsPerSecond = []
        countsPerSecondErr = []
        for i in range(len(stdSpectralCount)):
            countsPerSecond.append(stdSpectralCount[i] / exposureTime)
            countsPerSecondErr.append(stdSpectralCountErr[i] / exposureTime)
        # only have target because field did not solve
        targetInfo = open(subDir + '/target.txt', 'w')
        # initializes the wavelength to the far right of the spectra box
        wavelength = wavelengths[0][0]
        ratioIndex = 0
        # goes through each count and x value pair
        plotWvlen = []
        plotFlux = []
        plotFluxErr = []
        for i in range(len(countsPerSecond)):
            # only processes wavelengths which are in the true data
            if minWavelength <= wavelength <= maxWavelength:
                zeroAMcountCal = bestFitLine[ratioIndex][0][0]  # units of counts/second / flux / 1e12
                zeroAMcountCalErr = sqrt(abs(bestFitLine[ratioIndex][1][0][0]))  # units of counts/second / flux
                zeroAMcountCal *= 1e12                                           # / 1e12
                zeroAMcountCalErr *= 1e12
                zeroAMcount = countsPerSecond[i] / zeroAMcountCal
                try:
                    zeroAMcountErr = sqrt((countsPerSecondErr[i] ** 2 / countsPerSecond[i] ** 2 +
                                           zeroAMcountCalErr ** 2 / zeroAMcountCal ** 2)) * zeroAMcount
                except OverflowError:
                    zeroAMcount = float('nan')
                    zeroAMcountErr = float('nan')
                flux = zeroAMcount
                fluxErr = zeroAMcountErr
                flux /= wavelengths[i][0]  # convert back from total flux to flux per A
                fluxErr /= wavelengths[i][0]
                flux *= 10000  # convert from per A to per um
                fluxErr *= 10000
                ratioIndex += 1
                if isnan(flux):
                    targetInfo.write('# ')
                elif fluxErr / flux > 0.5:
                    targetInfo.write('# ')
                else:
                    plotWvlen.append(wavelength / 10)
                    plotFlux.append(flux)
                    plotFluxErr.append(fluxErr)
                targetInfo.write('Fl ' + objectID + '\t{:8.3f}'.format(wavelength / 10) +
                                 '\t{:5.4f}'.format(wavelengths[i][0] / 10) + '\t{:6.4e}'.format(flux) +
                                 '\t{:6.4e}'.format(fluxErr) + '\t# test data from 31-inch pipelineSP, run ' +
                                 strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
                targetInfo.write('\n')
            # increments wavelength either way
            wavelength = wavelengths[i + 1][0]
        targetInfo.close()
        plotFluxMedian = median(plotFlux)
        stdout.write('plotPoints {:5d}'.format(len(plotFlux)) + ' plotFluxMedian {:6.4e}'.format(plotFluxMedian) + '\n')
        if (len(plotWvlen)) > 0:
            popped = 0
            for j in range(len(plotFlux)):
                # noinspection PyTypeChecker
                if plotFlux[j - popped] > 5 * plotFluxMedian:
                    plotWvlen.pop(j - popped)
                    plotFlux.pop(j - popped)
                    plotFluxErr.pop(j - popped)
                    popped += 1
            plt.figure()
            #plt.plot(range(len(slopes)), slopes)
            #plt.scatter(range(len(slopes)), slopes, marker = 'o')
            plt.errorbar(plotWvlen, plotFlux, yerr=plotFluxErr, fmt='o')
            #plt.ylim(0,0.4)
            plt.savefig(subDir + '//' + objectID + '.png', bbox_inches='tight')
            plt.clf()
    imageData.close()
    clock = time() - clock
    totalTime = time() - startTime
    print("Elapsed Time: " + str(round(clock)) + " seconds (Total Time: " + str(round(totalTime)) + " seconds)")
