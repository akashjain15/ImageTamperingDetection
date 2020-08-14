import numpy as np
import sys


states, diseases, habbits,threshold = 0, 0, 0, 0
totalFeatures = 0
diseaseHabbitMat = []
totalData = []
statesVec = []
neighStates = []
resultMat = []


def readInputFile1(fileName):

    inputFile = open(fileName, 'r')

    lines = inputFile.readlines()

    if (len(lines) < 5):
        print("Wrong input file")
        exit()

    global states, diseases, habbits, totalFeatures, threshold

    states          = int(lines[0])
    diseases        = int(lines[1])
    habbits         = int(lines[2])
    threshold       = float(lines[3])
    totalFeatures   = diseases + habbits + 1

    matrix = lines[4:]

    if (len(matrix) != habbits):
        print("Invalid matrix, check input file")
        exit()

    global diseaseHabbitMat

    for line in matrix:
        values = line.split()

        if (len(values) != diseases):
            print("Invalid matrix, check input file")
            exit()

        habbitMat = []

        for val in values:
            status = int(val)
            habbitMat.append(status)

        diseaseHabbitMat.append(habbitMat)

    inputFile.close()


def readInputFile2(fileName):

    inputFile = open(fileName, 'r')

    lines = inputFile.readlines()

    global totalData, statesVec

    for line in lines:
        values = line.split()

        if (len(values) != totalFeatures):
            print("Invalid features, check input file")
            exit()

        data = []

        for value in values:
            val = float(value)
            data.append(val)

        totalData.append(data)
        statesVec.append(data[-1])

    statesVec.sort()
    inputFile.close()


def printResut(fileName, meanClusters):
    
    outFile = open(fileName, 'w+')

    for cluster in meanClusters:
        space = 0

        for i in range(0, totalFeatures):
            if (space == 0):
                space = 1
                outFile.write("%0.2f" % cluster[i])
            else:
                outFile.write(" %0.2f" % cluster[i])

        outFile.write("\n")

    outFile.close()

def initClusters():
    clusters = []

    noOfPts = len(statesVec)
    x = noOfPts / states

    for i in range(0, states):
        s = 0
        offset = i * x

        s = np.sum(statesVec[offset: offset+x])

        clusters.append(s/x)

    return clusters


def updateClusters(oldClusters):
    totalPtsInCluster = []
    clusters = []

    for i in range(0, states):
        totalPtsInCluster.append(0)
        clusters.append(0.0)


    for pt in statesVec:
        dist = []

        for cluster in oldClusters:
            dist.append(np.abs(cluster - pt))
            
        idx = dist.index(min(dist))
        totalPtsInCluster[idx] += 1
        clusters[idx] += pt

    for i in range(0, states):

        if (totalPtsInCluster[i] == 0):
            clusters[i] = oldClusters[i]
            continue

        clusters[i] = float(clusters[i] / totalPtsInCluster[i])

    return clusters


def findMeanClusters(clusters):
    totalPtsInCluster = []
    meanClusters = []

    for i in range(0, states):
        totalPtsInCluster.append(0)
        feat = []

        for j in range(0, totalFeatures):
            feat.append(0.0)

        meanClusters.append(feat)

    for feat in totalData:
        dist = []

        for cluster in clusters:
            dist.append(np.abs(cluster - feat[-1]))
            
        idx = dist.index(min(dist))
        totalPtsInCluster[idx] += 1

        for j in range(0, totalFeatures):
            meanClusters[idx][j] += feat[j]

    for i in range(0, states):

        if (totalPtsInCluster[i] == 0):
            continue

        for j in range(0, totalFeatures):
            meanClusters[i][j] = float(meanClusters[i][j] / totalPtsInCluster[i])

    return meanClusters

def clusterData():
    clusters = initClusters()

    oldClusters = []

    while (oldClusters != clusters):
        oldClusters = clusters
        clusters = updateClusters(oldClusters)

    return clusters

def main(argv):
    if (len(argv) < 4):
        print ("Less arguments")
        exit(0)

    readInputFile1(argv[1])
    readInputFile2(argv[2])
    clusters = clusterData()
    meanClusters = findMeanClusters(clusters)

    printResut(argv[3], meanClusters)

if __name__=='__main__':
    argumentList = sys.argv
    main(argumentList)

