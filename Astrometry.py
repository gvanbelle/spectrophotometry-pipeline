__author__ = 'ryan'

from Preprocessing import *
from multiprocessing import Process, Value, cpu_count
from os import chdir, system
from astropy import wcs
from numpy import array
from time import strptime, mktime
from math import acos, degrees, asin, sin, radians, cos


def astrometry(subDir, solvedCount, XSum, YSum):
    clock = time()
    print("Running Astrometry.net on " + subDir)
    chdir(subDir)
    fitsFile = ''
    for file in listdir(subDir):
        if file.endswith('.fits'):
            fitsFile = file
    # runs astrometry.net on the frames
    hdulist = fits.open(fitsFile)
    telra = hdulist[0].header['TELRA'].strip()
    teldec = hdulist[0].header['TELDEC'].strip()
    telha = hdulist[0].header['TELHA'].strip()
    name = hdulist[0].header['OBJECT'].strip()
    AM = hdulist[0].header['AIRMASS']
    hdulist.close()
    run = False
    for file in listdir(subDir):
        if file.endswith('Info.txt'):
            run = True
    if not run:
        system('/usr/local/astrometry/bin/solve-field --scale-units arcsecperpix --scale-low 0.4525 --scale-high 0.4600'
               + ' --ra ' + telra + ' --dec ' + teldec + ' --radius 0.75 ' + fitsFile)  # + ' --no-plots')
    solved = False
    for file in listdir(subDir):
        if file.endswith('.solved'):
            solved = True
    # writes the Info files
    info = open(subDir + '/Info.txt', 'w')
    info.write('StarID: ' + name)
    RA = starNamesListNameRADE[1][starNamesListNameRADE[0].index(name)]
    info.write('\nRight Ascension: ')
    if RA != '0.0':
        info.write(RA)
    else:
        info.write('No SIMBAD result')
        solved = False
    DE = starNamesListNameRADE[2][starNamesListNameRADE[0].index(name)]
    info.write('\nDeclination: ')
    if DE != '0.0':
        info.write(DE)
    else:
        info.write('No SIMBAD result')
    if solved:
        RAHour = int(RA[:2])
        RAMin = int(RA[3:5])
        RASec = float(RA[6:])
        RADeg = RAHour * 15 + RAMin / 4 + RASec / 240
        DEDegree = int(DE[1:3])
        DEArcMin = int(DE[4:6])
        DEArcSec = float(DE[7:])
        DEDeg = DEDegree + DEArcMin / 60 + DEArcSec / 3600
        if DE[0] == '-':
            DEDeg *= -1
        if starNamesListPMNameRADE[1][starNamesListPMNameRADE[0].index(name)] == '--':
            PMRA = 0.0
            PMDEC = 0.0
        else:
            PMRA = float(starNamesListPMNameRADE[1][starNamesListPMNameRADE[0].index(name)]) / 1000
            PMDEC = float(starNamesListPMNameRADE[2][starNamesListPMNameRADE[0].index(name)]) / 1000
        years = (time() - mktime(strptime('Jan 1 00', '%b %d %y'))) / 60 / 60 / 24 / 365.25
        adjRA = RADeg + PMRA * years / 3600
        adjDE = DEDeg + PMDEC * years / 3600
        HA = int(telha[1:3]) * 15 + int(telha[4:6]) / 60 + int(telha[7:]) / 3600
        if telha[0] == 'E':
            HA *= -1
        ZA = degrees(acos(1 / AM))
        info.write('\nZenith Angle: ' + str(ZA))
        ALT = degrees(asin(sin(radians(adjDE)) * sin(radians(obsLatDeg)) +
                           cos(radians(adjDE)) * cos(radians(obsLatDeg)) * cos(radians(HA))))
        A = degrees(acos((sin(radians(adjDE)) - sin(radians(ALT)) * sin(radians(obsLatDeg))) /
                         (cos(radians(ALT)) * cos(radians(obsLatDeg)))))
        AZ = 0
        if sin(radians(HA)) < 0:
            AZ = A
        else:
            AZ = 360 - A
        info.write('\nAzimuth: ' + str(AZ))
        wcsFile = ''
        for file in listdir(subDir):
            if file.endswith('.wcs'):
                wcsFile = subDir + '/' + file
        wcsFile = fits.open(wcsFile)
        wcsInfo = wcs.WCS(wcsFile[0].header)
        RADEtoXY = wcsInfo.wcs_world2pix(array([[adjRA, adjDE]]), 1)
        info.write('\nX: ' + str(RADEtoXY[0][0]) + '\nY: ' + str(RADEtoXY[0][1]))
        solvedCount.value += 1
        XSum.value += RADEtoXY[0][0]
        YSum.value += RADEtoXY[0][1]
    else:
        info.write('\nNo Solution')
    info.close()
    clock = time() - clock
    totalTime = time() - startTime
    print("Elapsed Time: " + str(round(clock)) + " seconds (Total Time: " + str(round(totalTime)) + " seconds)")

solvingProcs = []
solvedCount = Value('d', 0.0)
cpuCount = cpu_count() - 1
XSum = Value('d', 0.0)
YSum = Value('d', 0.0)
obsLatDeg = 35.0968722222
for subDir in chain(trimmedCalibrationFrames, trimmedScienceFrames):
    if len(solvingProcs) < cpuCount:
        p = Process(target=astrometry, args=(subDir, solvedCount, XSum, YSum))
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

if solvedCount.value:
    XAvg = round(XSum.value / solvedCount.value)
    YAvg = round(YSum.value / solvedCount.value)
else:
    XAvg = 1337
    YAvg = 848
