# Author: Sudnya Padalikar
# Date  : Dec 23 2013
# Brief : A python script to call various classifiers implemented in scikit learn
# Comment : I will try to abide by https://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Python_Style_Rules

#!/usr/bin/python
import sys
import argparse
import couchdb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

maxTrainingIterations = 800
maxTestIterations = maxTrainingIterations/10

# reads in the feature schema from the variables.txt - assume there are multiple categories and each category has many classes
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



# creates labels from all the available categories and their subclasses
def extractLabels(document, featureList):
    labels = {}
    for i, j in featureList.iteritems():
        for k in range(j):
            labels[i+str(k+1)] = document[i + str(k + 1)]
    return labels


# to re-baseline values between -1 and 1
def standardize(character):
    return (float(ord(character)) / 127.0) - 1.0



# creates a feature extractor that will generate the features from samples (instead of char->float)
def generateFeatureExtractor(corpus):
    featureExtractor = CountVectorizer()
    featureExtractor.fit(corpus)
    return featureExtractor



# converts text/image data into numeric feature usable by ML algorithms
def extractFeatures(featureExtractor, samples):
    features = featureExtractor.transform(samples)
    return features


# this was written initially for a naive implementation that converted char->float as features
def extractRawFeatures(text):
    features = []
    for i in text:
        features.append(standardize(i))
    while len(features) < 180:
        features.append(0.0)
    return features



# k1, k2, s1, s2 etc
def getSampleLabels(sampleLabels, i):
    col = []
    for j in sampleLabels:
        col.append(float(j[i]))
    return col



# to handle useless labels
def noSamplesHaveThisLabel(sampleLabels, i):
    noLabelOn = True
    for j in sampleLabels:
        if float(j[i]) != 0:
            return False
    return noLabelOn



# which class in a specified category has most confidence
def findMaxClass(output):
    maxInCategories = {}
    maxTypeInCategory = {}

    for label, value in output.iteritems():
        if not label[0] in maxInCategories:
            maxInCategories[label[0]] = value
            maxTypeInCategory[label[0]] = label
        elif value > maxInCategories[label[0]]:
            maxInCategories[label[0]] = value
            maxTypeInCategory[label[0]] = label
    return maxTypeInCategory



# to write output as expected by kaggle's sample submission format
classesInOrder = [ 's1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
def writeTweetPrediction(outputFile, id, tweetClassifications):
    resultString = str(id)

    for possibleClass in classesInOrder:
        resultString += "," + str(tweetClassifications[possibleClass])

    outputFile.write(resultString + "\n")



# open couchdb at a given IP:port - change this as per your machine
def openDb():
    couch = couchdb.Server('http://192.168.1.106:5984/')
    return couch



# classfier function
def classify(dbname, method):
    totalTypes, typeMap = readVariablesInMap()
    featureList = {}
    for t, f in typeMap.iteritems():
        featureList[t] = len(f)

    couch = openDb()
    db = couch[dbname]
    
    # each sample is an array of features
    samples = []
    # each sample is an array of labels
    sampleLabels = []
    count = 0

    for tweet in db.iterview('_all_docs', maxTrainingIterations, include_docs=True):
        # x (features)
        tweetText = tweet.doc['tweet']
        # y class labels (1 hot encoded)
        tweetLabels = extractLabels(tweet.doc, featureList)
        samples.append(tweetText)
        sampleLabels.append(tweetLabels)
        if count > maxTrainingIterations:
            break
        count += 1


    # geneate feature selector
    featureExtractor = generateFeatureExtractor(samples)
    svms = {}
    sparseSamples = extractFeatures(featureExtractor, samples)

    for labelType in totalTypes:
        print "Training svm for label ", str(labelType)
        clf = svm.LinearSVC(class_weight='auto')
        if not noSamplesHaveThisLabel(sampleLabels, labelType):
            thisSampleLabel = getSampleLabels(sampleLabels, labelType)
            clf.fit(sparseSamples, thisSampleLabel)
            svms[labelType] = clf
        # for small subset of training data, we could have an empty SVM for a specific label
        else:
            svms[labelType] = None


    #cross validate
    cvDbName = 'cross-validation-tweets' 
    cvDb = couch[cvDbName]
    ctr = 0
    matches = 0
    totalSamples = 0
    
    for i in cvDb.iterview('_all_docs', maxTrainingIterations, include_docs=True):
        cvTweet = i.doc['tweet']
        cvSample = extractFeatures(featureExtractor, [cvTweet])
        outputs = {}
        for label, currSvm in svms.iteritems():
            if currSvm != None:
                current = currSvm.predict(cvSample)
                outputs[label] = current[0]
            else:
                outputs[label] = 0.0
        maxInEachCategory = findMaxClass(outputs)

        for category, labelsInCategory in typeMap.iteritems():
            if float(i.doc[maxInEachCategory[category]]) > 0.0:
                matches += 1
            totalSamples += 1
        if ctr > maxTrainingIterations:
            break
        ctr += 1


    outputFile = open('submission.csv', 'w')
    outputFile.write('id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15\n')
    #test
    testDbName = 'test-tweets' 
    testDb = couch[testDbName]
    ctr = 0
    for i in testDb:
        testTweet = testDb[i]['tweet']
        testSample = extractFeatures(featureExtractor, [testTweet])
        outputs = {}
        for label, currSvm in svms.iteritems():
            if currSvm != None:
                current = currSvm.predict(testSample)
                outputs[label] = current[0]
            else:
                outputs[label] = 0.0
        maxInEachCategory = findMaxClass(outputs)
        writeTweetPrediction(outputFile, testDb[i]['id'], outputs)
        shouldPrint = True
        if ctr > maxTestIterations:
            shouldPrint = False
            break
        ctr += 1

        if shouldPrint:
            print testTweet
        for category, labelsInCategory in typeMap.iteritems():
            if shouldPrint:
                print category + ' = ' + labelsInCategory[maxInEachCategory[category]] + ' '

    print "Cross validation accuracy: ", str((matches * 100.0) / totalSamples)
    print "Cross validation RMSE:     ????"



def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-db',     help="name of the database which contains training data", type=str)
    parser.add_argument('-method', help="the machine learning method/algorithm to invoke. Currently supports: svm, naive-bayes",   type=str)
    args = parser.parse_args()
    classify(args.db, args.method);

if __name__ == '__main__':
    main()
