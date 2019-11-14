'''
Created on Nov 11, 2013

@author: Todd
'''
import numpy as np
from math import *
from collections import OrderedDict
from scipy import misc
import matplotlib.pyplot as plt
import cProfile
import pstats
import Image
from scipy import stats

def domainGet(matrix, domainIndex):
    dim = matrix.shape[0]
    divisor = dim/2 -1
    row = 2*(domainIndex/divisor)
    col = 2*(domainIndex%divisor)
    domainBlock = matrix[row: row + 4, col: col+4]
    if domainBlock.shape == (4,4):
        return domainBlock

def rangeGet(matrix, rangeIndex):
    dim = matrix.shape[0]
    divisor = (dim/2)
    row = 2*(rangeIndex/divisor)
    col = 2*(rangeIndex%divisor)
    rangeBlock = matrix[row: row+2, col:col+2]
    if rangeBlock.shape == (2,2):
        return rangeBlock   

def transform(matrix, domainIndex):
    #compresses a domain block into a range block
    domainBlock = domainGet(matrix, domainIndex)
    r1 = domainBlock[0:2, 0:2]
    r2 = domainBlock[0:2, 2:4]
    r3 = domainBlock[2:4, 0:2]
    r4 = domainBlock[2:4, 2:4]
    r1average = np.average(r1)
    r2average = np.average(r2)
    r3average = np.average(r3)
    r4average = np.average(r4)
    newRangeBlock = np.matrix([[np.uint8(r1average), np.uint8(r2average)], [np.uint8(r3average), np.uint8(r4average)]])
    return newRangeBlock

def identity(matrix):
    return matrix

def rotate90(matrix):
    rotationMatrix = np.matrix([[0, -1],[1, 0]])
    return rotationMatrix*matrix

def rotate180(matrix):
    rotationMatrix = np.matrix([[-1, 0],[0, -1]])
    return rotationMatrix*matrix

def rotate270(matrix):
    rotationMatrix = np.matrix([[0, 1],[-1, 0]])
    return rotationMatrix*matrix

def reflectY(matrix):
    reflectionMatrix = np.matrix([[-1,0], [0,1]])
    return reflectionMatrix*matrix

def reflectX(matrix):
    reflectionMatrix = np.matrix([[1,0], [0,-1]])
    return reflectionMatrix*matrix

def reflectMainDiag(matrix):
    reflectionMatrix = np.matrix([[0,1],[1,0]])
    return reflectionMatrix*matrix

def reflectOppDiag(matrix):
    reflectionMatrix = np.matrix([[0,-1],[-1,0]])
    return reflectionMatrix*matrix

def createTransformationDictionary(matrix):
    #creates a dictionary with the values being all of the possible transformations
    transformationDictionary = {0: identity(matrix), 1: rotate90(matrix), 2:rotate180(matrix),
                                3:rotate270(matrix), 4:reflectX(matrix), 5:reflectY(matrix),
                                6:reflectMainDiag(matrix), 7: reflectOppDiag(matrix)}
    return transformationDictionary

def variance(matrix, index, blocksize):
    #calculates the variance of a domain or range block
    if blocksize == 4:
        block = domainGet(matrix, index)
    if blocksize == 2:
        block = rangeGet(matrix,index)
    totalvariance = np.var(block)
    return totalvariance

def createDomainDictionary(matrix):
    #creates a dictionary with keys being the domain index and the values
    #being the variance of each of those domain blocks
    size = (((matrix.shape[0])/2)-1)**2
    domainDict = {i: variance(matrix, i, 4) for i in range(size)}
    return domainDict

def createRangeDictionary(matrix):
    #creates a dictionary with keys being the range index and the values
    #being the variance of each of those range blocks
    size = (matrix.shape[0]**2)/4
    rangeDict = {i: variance(matrix,i,2) for i in range(size)}
    return rangeDict

def domainSorting(matrix):
    #sorts the domainDictionary by variance
    domainDict = createDomainDictionary(matrix)
    sortedDomainDict = OrderedDict(sorted(domainDict.items(), key = lambda x:x[1]))
    return sortedDomainDict

def rangeSorting(matrix):
    #sorts the rangeDictionary by variance, highest to lowest
    rangeDict = createRangeDictionary(matrix)
    sortedRangeDict = OrderedDict(sorted(rangeDict.items(), key = lambda x:x[1]))
    return sortedRangeDict

def grouping(matrix):
    #matches the domain and range blocks based on their variances
    rangeIndexList = rangeSorting(matrix).keys()
    domainIndexList = domainSorting(matrix).keys()
    domainTotal = len(domainIndexList)
    rangeTotal = len(rangeIndexList)
    groupedDict = {rangeIndexList[i]: (domainIndexList[(domainTotal*i)/rangeTotal]) for i in range(rangeTotal)}
    return groupedDict

def createMultipleCompressedDictionary(matrix):
    #creates a dictionary with the values being all possible
    #transformations of a domain block with the key being the index
    groupedDict = grouping(matrix)
    rangeTotal = (matrix.shape[0]**2)/4
    compressedMultipleDictionary = {}
    for ri in range(rangeTotal):
        domainIndex = groupedDict.get(ri)
        transformationDict = createTransformationDictionary(transform(matrix, domainIndex))
        transformations = transformationDict.values()
        compressedMultipleDictionary[ri] = transformations
    return compressedMultipleDictionary 

def varianceDifference(compressedDomainBlock, rangeBlock):
    #calculates the variance between the compressed domain block and
    #the range block it is replacing
    differenceBlock = compressedDomainBlock-rangeBlock
    totalvariance = np.var(differenceBlock)
    return totalvariance

def createVarianceDict(matrix):
    #compares the variances of each possible transformation of the
    #domain block with the correct range block
    compressedDomainDictionary = createMultipleCompressedDictionary(matrix)
    rangeIndexList = rangeSorting(matrix).keys()
    rangeTotal = (matrix.shape[0]**2)/4
    varianceDict = {index: ([varianceDifference(item, rangeGet(matrix,rangeIndexList[index]))
                             for item in compressedDomainDictionary.get(index)]) for index in range(rangeTotal)}
    return varianceDict

def leastVarianceTransformation(matrix):
    #creates a dictionary of which transformation produces the
    #smallest variation for a given index
    varianceDict = createVarianceDict(matrix)
    transformationDict = {key: (varianceDict.get(key)).index(min(varianceDict.get(key))) for key in varianceDict}
    return transformationDict

def createCompressedDictionary(matrix):
    #creates a dictionary of matrices that correspond to the
    #smallest possible variances for a given range index
    groupedDict = grouping(matrix)
    transformationDict = leastVarianceTransformation(matrix)
    rangeTotal = (matrix.shape[0]**2)/4
    compressedDomainDictionary = {ri: createTransformationDictionary(
        transform(matrix, groupedDict.get(ri))).get(transformationDict[ri]) for ri in range(rangeTotal)}
    return compressedDomainDictionary

def scaling(matrix):
    #looks at the max and min of the best domain block and its range block
    #calculates slope and y-intercept for scaling
    scaledDictionary = {}
    compressedDomainDictionary = createCompressedDictionary(matrix)
    for key in compressedDomainDictionary:
        x0 = np.float64(np.amin(compressedDomainDictionary[key]))
        x1 = np.float64(np.amax(compressedDomainDictionary[key]))
        y0 = np.float64(np.amin(rangeGet(matrix, key)))
        y1 = np.float64(np.max(rangeGet(matrix, key)))
        if x1-x0 == 0.0:
            m = 0
            b = np.float64(np.average(rangeGet(matrix,key)))
        else:
            m = np.float64((y1-y0))/np.float64((x1-x0))
            b = y1-(m*x1)
        scaledDictionary[key] = (m,b,(compressedDomainDictionary[key]*m)+b)
        #print x0, x1, y0, y1, y1-y0, x1-x0, m, b, scaledDictionary[key]
    return scaledDictionary

def scaling2(matrix):
    #looks at the least squares linear regression for scaling
    scaledDictionary = {}
    compressedDomainDictionary = createCompressedDictionary(matrix)
    for key in compressedDomainDictionary:
        domainBlock = np.array(compressedDomainDictionary[key]).flatten()
        domainBlock2 = np.array([domainBlock, np.ones(len(domainBlock))])
        rangeBlock = np.array(rangeGet(matrix, key)).flatten()
        w = np.linalg.lstsq(domainBlock2.T,rangeBlock)[0]
        scaledDictionary[key] = (np.float64(w[0]),np.float64(w[1]),(np.float64(w[0])*compressedDomainDictionary[key])+np.float64(w[1]))
    return scaledDictionary

def createIFSDictionary(matrix):
    #creates a dictionary with the keys being the range index and the values being an array of the transformations used to get to the compressed image.
    transformationDictionary = leastVarianceTransformation(matrix)
    IFSDictionary = {key: transformationDictionary.get(key) for key in transformationDictionary}
    return IFSDictionary

def decompress(matrix, image):
    imageMatrix = np.matrix(image)
    dim = imageMatrix.shape[0]
    divisor = (dim/2)
    IFSDictionary = createIFSDictionary(matrix)
    scalingDictionary = scaling2(matrix)
    for key, value in IFSDictionary.items():
        row = 2*(key/divisor)
        col = 2*(key%divisor)
        rangeBlock = rangeGet(imageMatrix, key)
        m = scalingDictionary[key][0]
        b = scalingDictionary[key][1]
        transformationDictionary = createTransformationDictionary(rangeBlock)
        imageMatrix[row: row+2, col:col+2] = (m*(transformationDictionary[value]))+b
    return imageMatrix

def decompress2(matrix):
    dim = matrix.shape[0]
    divisor = (dim/2)
    IFSDictionary = createIFSDictionary(matrix)
    scalingDictionary = scaling2(matrix)
    for key, value in IFSDictionary.items():
        row = 2*(key/divisor)
        col = 2*(key%divisor)
        m = scalingDictionary[key][0]
        b = scalingDictionary[key][1]
        rangeBlock = scalingDictionary[key][2]
        transformationDictionary = createTransformationDictionary(rangeBlock)
        #matrix[row: row+2, col:col+2] = (m*(transformationDictionary[value]))+b
        matrix[row: row+2, col:col+2] = rangeBlock
    return matrix

def decompressLoop(matrix, image):
    varDif = 1
    varDif2 = 0
    i=0
    while i < 10:
        oldImage = image
        imageMid = decompress(matrix, oldImage)
        varDif = varianceDifference(imageMid,oldImage)
        image = decompress(matrix, imageMid)
        varDif2 = varianceDifference(image, imageMid)
        print varDif-varDif2
        i=i+1
    return image

def decompressLoop2(matrix):
    image = matrix
    varDif = 1
    varDif2 = 0
    i=0
    while i < 10:
        print i
        if i == 0:
            #print image
            imageMid = decompress2(matrix)
            #print
            #print imageMid
            varDif = varianceDifference(imageMid,image)
            image = decompress(matrix, imageMid)
            #print
            #print image
            varDif2 = varianceDifference(image, imageMid)
            #print varDif-varDif2
            i=i+1
        else:
            oldImage = image
            #print oldImage
            imageMid = decompress(matrix, oldImage)
            #print
            #print imageMid
            varDif = varianceDifference(imageMid,oldImage)
            image = decompress(matrix, imageMid)
            #print 
            #print image
            varDif2 = varianceDifference(image, imageMid)
            #print varDif-varDif2
            i=i+1
    return image
    
def createImageMatrix():
    #creates the lena matrix
    lena = misc.lena()
    lenaMatrix = np.matrix(lena)
    return lenaMatrix

def createSmallLenaMatrix():
    #a 32 by 32 version of the lena matrix
    lena = misc.imresize(createImageMatrix(), .0625)
    lenaMatrix = np.matrix(lena)
    return lenaMatrix

def showNewImage(matrix):
    #shows the new matrix 
    plt.imshow(matrix, cmap=plt.cm.gray)
    plt.show()

def createTest1():
    block = np.asarray(Image.open('test1small.png').convert("L"))
    blockMatrix = np.matrix(block)
    return blockMatrix

def createWhiteTest():
    block = np.asarray(Image.open('whiteTest.png').convert("L"))
    blockMatrix = np.matrix(block)
    return blockMatrix


showNewImage(decompressLoop2(createImageMatrix()))


