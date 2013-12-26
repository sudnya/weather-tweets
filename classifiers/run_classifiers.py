# Author: Sudnya Padalikar
# Date  : Dec 23 2013
# Brief : A python script to call various classifiers implemented in scikit learn
# Comment : I will try to abide by https://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Python_Style_Rules

#!/usr/bin/python
import sys
import argparse
import couchdb
from sklearn import svm



def readVariablesInMap():
    types = []
    varMap = {}
    varFile = open('data/variableNames.txt', 'r')
    for line in varFile:
        if '===' in line:
            continue
        if '=' in line.split():
            if not line.split()[0] in varMap:
                varMap[line.split()[0]] = {}
            continue
        else:
            varMap[line[0]][line.partition(',')[0]] = line.partition(',')[2].strip()
            types.append(line.partition(',')[0])
    return types, varMap



def extractLabels(document, featureList):
    labels = {}
    for i, j in featureList.iteritems():
        for k in range(j):
            labels[i+str(k+1)] = document[i + str(k + 1)]
    return labels

def standardize(character):
    return (float(ord(character)) / 127.0) - 1.0

def extractFeatures(text):
    features = []
    for i in text:
        features.append(standardize(i))
    while len(features) < 180:
        features.append(0.0)
    return features



def getLabelCount(samples):
    return len(samples[0])



def getSampleLabels(sampleLabels, i):
    col = []
    for j in sampleLabels:
        col.append(float(j[i]))
    #print col
    return col

def noSamplesHaveThisLabel(sampleLabels, i):
    noLabelOn = True
    for j in sampleLabels:
        if float(j[i]) != 0:
            return False
    return noLabelOn

def isPredictionThisClass(output, idx):
    if output[idx-1] == 1.0:
        return True
    return False

def findMaxClass(output):
    maxInCategories = {}
    maxTypeInCategory = {}


    for label, value in output.iteritems():
        #print ' id = ' + str(id) + ' which contains i = ' + str(i) + '\n'
        if not label[0] in maxInCategories:
            maxInCategories[label[0]] = value
            maxTypeInCategory[label[0]] = label
        elif value > maxInCategories[label[0]]:
            
            maxInCategories[label[0]] = value
            maxTypeInCategory[label[0]] = label

    return maxTypeInCategory

maxIterations = 1000

def svmClassify(dbname):
    totalTypes, typeMap = readVariablesInMap()
    
    featureList = {}
    for t, f in typeMap.iteritems():
        featureList[t] = len(f)

    print 'Opening couchdb named ' + dbname
    couch = couchdb.Server('http://192.168.1.106:5984/')
    db = couch[dbname]

    # array of samples, each sample is an array of features
    samples = []

    # array of samples, each sample is an array of labels
    sampleLabels = []

    count = 0
    for id in db:
        # x (features)
        tweetText = db[id]['tweet']

        # y class labels (1 hot encoded)
        tweetLabels = extractLabels(db[id], featureList)

        samples.append(extractFeatures(tweetText))
        sampleLabels.append(tweetLabels)
#        if count > maxIterations:
#            break
#        count += 1

    svms = {}

    for labelType in totalTypes:
        print "training svm for label ", str(labelType)
        clf = svm.LinearSVC(class_weight='auto')

        if not noSamplesHaveThisLabel(sampleLabels, labelType):
            thisSampleLabel = getSampleLabels(sampleLabels, labelType)
            #print thisSampleLabel
            clf.fit(samples, thisSampleLabel)
            svms[labelType] = clf
        else:
            svms[labelType] = None

    #test
    testDbName = 'test-tweets' 
    testDb = couch[testDbName]
    ctr = 0
    for i in testDb:
        testTweet = testDb[i]['tweet']
        testSample = extractFeatures(testTweet)
        #print 'Feeding in testSample of ' + str(testSample)
        outputs = {}
        for label, currSvm in svms.iteritems():
            if currSvm != None:
                #print 'Current SVM = ' + str(currSvm.support_vectors_)
                current = currSvm.predict([testSample])
                #print 'Non-zero feature predicted : ' + str(current)
                outputs[label] = current[0]
            else:
                outputs[label] = 0.0
        maxInEachCategory = findMaxClass(outputs)

        print testTweet
        for category, labelsInCategory in typeMap.iteritems():
            print category + ' = ' + labelsInCategory[maxInEachCategory[category]] + ' '

        if ctr > 1000:
            break
        ctr += 1




def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-db',     help="name of the database which contains training data", type=str)
    parser.add_argument('-method', help="the machine learning method/algorithm to invoke",   type=str)
    args = parser.parse_args()
    if (args.method == "svm"):
        svmClassify(args.db);

if __name__ == '__main__':
    main()
