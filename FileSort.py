__author__ = 'ryan'

from sys import argv
from os.path import exists
from os import listdir
from astropy.io import fits
from astroquery.simbad import Simbad

# sorts the files in the given directory
if len(argv) < 3:
    print(argv[0] + ' <directory> <filter>')
    exit()
directory = argv[1]
if not exists(directory):
    print('Directory does not exist')
    exit()
filterType = argv[2].upper()
fitsFiles = filter(lambda x: x.endswith('.fits'), listdir(directory))
if not directory.endswith('/'):
    directory += '/'
filterFiles = []
for file in fitsFiles:
    file = directory + file
    hdulist = fits.open(file)
    if hdulist[0].header['FILTER1'].strip() == filterType or hdulist[0].header['FILTER2'].strip() == filterType:
        filterFiles.append(file)
    hdulist.close()
objectFiles = []
flatDarkBiasFiles = []
for file in filterFiles:
    hdulist = fits.open(file)
    if hdulist[0].header['IMAGETYP'].strip() == 'object':
        objectFiles.append(file)
    else:
        flatDarkBiasFiles.append(file)
    hdulist.close()
if not exists(directory + 'star_names_list_pm.dat'):
    starNamesListPM = open(directory + 'star_names_list_pm.dat', 'w')
    simbad = Simbad()
    simbad.add_votable_fields('pm')
    newList = False
    if not exists(directory + 'star_names_list.dat'):
        starNamesList = open(directory + 'star_names_list.dat', 'w')
        newList = True
    for file in objectFiles:
        hdulist = fits.open(file)
        objectID = hdulist[0].header['OBJECT'].strip()
        objectQuery = simbad.query_object(objectID)
        if objectQuery is None:
            if newList:
                # noinspection PyUnboundLocalVariable
                starNamesList.write(objectID + ' | ' + hdulist[0].header['TELRA'].strip() + ' | ' +
                                    hdulist[0].header['TELDEC'].strip() + ' | WCS | auto\n')
            starNamesListPM.write(objectID + ' | 0 | 0\n')
        else:
            if newList:
                # noinspection PyUnboundLocalVariable
                starNamesList.write(objectID + ' | ' + objectQuery[0]['RA'] + ' | ' + objectQuery[0]['DEC'] +
                                    ' | WCS | auto\n')
            starNamesListPM.write(objectID + ' | ' + str(objectQuery[0]['PMRA']) + ' | ' + str(objectQuery[0]['PMDEC'])
                                  + '\n')
        hdulist.close()
    if newList:
        starNamesList.close()
    starNamesListPM.close()
starNamesList = open(directory + 'star_names_list.dat', 'r')
starNamesListNameRADE = [[], [], []]
for line in starNamesList.readlines():
    for i, item in enumerate(line.split('|')):
        if i < 3:
            # noinspection PyTypeChecker
            starNamesListNameRADE[i].append(item.strip())
starNamesList.close()
starNamesListPM = open(directory + 'star_names_list_pm.dat', 'r')
starNamesListPMNameRADE = [[], [], []]
for line in starNamesListPM.readlines():
    for i, item in enumerate(line.split('|')):
        # noinspection PyTypeChecker
        starNamesListPMNameRADE[i].append(item.strip())
starNamesListPM.close()
calibrationSourceNames = []
if filterType == 'TG':
    calibrationSourceNames = open('calibration source names.csv', 'r').read().splitlines()
else:
    calibrationData = open('Photometry Calibrators.csv', 'r')
    filterDesignation = calibrationData.readline().split('\t')
    filterID = calibrationData.readline().split('\t')
    calibrators = []
    for i, line in enumerate(calibrationData.readlines()):
        calibrators.append(line.split('\t'))
        calibrationSourceNames.append(calibrators[i][0])
    filterIndex = 0
    for i, label in enumerate(filterID):
        if label == filterType:
            filterIndex = i
calibrationFrames = []
calibrationObjects = []
scienceFrames = []
for file in objectFiles:
    hdulist = fits.open(file)
    objectID = hdulist[0].header['OBJECT'].strip()
    if objectID in starNamesListNameRADE[0]:
        if objectID in calibrationSourceNames:
            calibrationFrames.append(file)
            if objectID not in calibrationObjects:
                calibrationObjects.append(objectID)
        else:
            scienceFrames.append(file)
    hdulist.close()
#calibrator = max(calibrationObjects, key=calibrationObjects.count)
#sorting = True
#while sorting:
    #for i, objectID in enumerate(calibrationObjects):
        #if objectID != calibrator:
            #calibrationObjects.remove(objectID)
            #scienceFrames.append(calibrationFrames[i])
            #calibrationFrames.remove(calibrationFrames[i])
            #break
        #elif i == len(calibrationObjects) - 1:
            #sorting = False
            #break
for frame in calibrationFrames:
    scienceFrames.append(frame)