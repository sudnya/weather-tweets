# Author: Sudnya Padalikar
# Date  : Dec 23 2013
# Brief : A python script to call various classifiers implemented in scikit learn
# Comment : I will try to abide by https://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Python_Style_Rules

#!/usr/bin/python
import sys
import argparse
import couchdb
from math import sqrt
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

maxTrainingIterations = 1000
maxCrossValidateIterations = 1000
maxTestIterations = 1000000 #maxTrainingIterations/10

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



# creates a feature extractor that will generate the features from samples (instead of char->float)
def generateFeatureExtractor(corpus):
    #featureExtractor = TfidfVectorizer(max_features=None)# CountVectorizer()
    featureExtractor = CountVectorizer()
    featureExtractor.fit(corpus)
    return featureExtractor



# converts text/image data into numeric feature usable by ML algorithms
def extractFeatures(featureExtractor, samples):
    features = featureExtractor.transform(samples)
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



# normalize the output for each category
def normalizeOutput(classification):
    sums = {}
    for i, val in classification.iteritems():
        if not i[0] in sums:
            sums[i[0]] = 0
        
        sums[i[0]] += val

    for key, val in classification.iteritems():
        if sums[key[0]] == 0.0:
            classification[key] = 0.0
        else:
            classification[key] = val/sums[key[0]]

    return classification

# to write output as expected by kaggle's sample submission format
classesInOrder = [ 's1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
def writeTweetPrediction(outputFile, id, tweetClassifications):
    resultString = str(id)

    for possibleClass in classesInOrder:
        resultString += "," + str(tweetClassifications[possibleClass])

    outputFile.write(resultString + "\n")



# handles cross validate set - running on SVM, calculating Root mean sqr error calculation
def crossValidate(cvDb, maxIterations, featureExtractor, svms, typeMap):
    ctr = 0
    matches = 0
    totalSamples = 0
    predictedOutputs = []
    expectedOutputs = []
    
    for i in cvDb.iterview('_all_docs', maxIterations, include_docs=True):
        cvTweet = i.doc['tweet']
        cvTweet += i.doc['state']
        cvTweet += i.doc['location']
        outputs = {}

        cvSample = extractFeatures(featureExtractor, [cvTweet])

        for label, currSvm in svms.iteritems():
            if currSvm != None:
                current = currSvm.predict(cvSample)
                outputs[label] = current[0]
                #predictedOutputs.append(current[0])
            else:
                outputs[label] = 0.0
                #predictedOutputs.append(0.0)
        
            
            yRef = float(i.doc[label])
            expectedOutputs.append(yRef)

        outputs = normalizeOutput(outputs)
        predictedOutputs += [outputs[label] for label, svm in svms.iteritems()]
        if ctr > maxIterations:
            break
        ctr += 1
    print "Mean squared error = ", sqrt(mean_squared_error(expectedOutputs, predictedOutputs))


# open couchdb at a given IP:port - change this as per your machine
def openDb():
    couch = couchdb.Server('http://192.168.1.106:5984/')
    return couch



def test(testDb, maxTestIterations, featureExtractor, svms, typeMap, outputFileName):
    outputFile = open(outputFileName, 'w')
    outputFile.write('id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15\n')
    ctr = 0
    for i in testDb.iterview('_all_docs', maxTestIterations, include_docs=True):
        testTweet = i.doc['tweet']
        testTweet += i.doc['state']
        testTweet += i.doc['location']

        testSample = extractFeatures(featureExtractor, [testTweet])
        outputs = {}
        for label, currSvm in svms.iteritems():
            if currSvm != None:
                current = currSvm.predict(testSample)
                outputs[label] = current[0]
            else:
                outputs[label] = 0.0
        #maxInEachCategory = findMaxClass(outputs)
        outputs = normalizeOutput(outputs)
        writeTweetPrediction(outputFile, i.doc['id'], outputs)
        shouldPrint = True
        if ctr > maxTestIterations:
            shouldPrint = False
            break
        ctr += 1



# classfier function
def classify(dbname, method, outputFileName, trainingIterations):
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

    for tweet in db.iterview('_all_docs', trainingIterations, include_docs=True):
        # x (features)
        tweetText = tweet.doc['tweet']
        tweetText += tweet.doc['state']
        tweetText += tweet.doc['location']
        # y class labels (1 hot encoded)
        tweetLabels = extractLabels(tweet.doc, featureList)
        samples.append(tweetText)
        sampleLabels.append(tweetLabels)
        if count > trainingIterations:
            break
        count += 1


    # geneate feature selector
    featureExtractor = generateFeatureExtractor(samples)
    svms = {}
    sparseSamples = extractFeatures(featureExtractor, samples)

    for labelType in totalTypes:
        print "Training svm for label ", str(labelType)
        clf = svm.LinearSVC(class_weight='auto', C=20.0)
        if not noSamplesHaveThisLabel(sampleLabels, labelType):
            thisSampleLabel = getSampleLabels(sampleLabels, labelType)
            clf.fit(sparseSamples, thisSampleLabel)
            svms[labelType] = clf
        # for small subset of training data, we could have an empty SVM for a specific label
        else:
            svms[labelType] = None


    #cross validate
    cvDb = couch['cross-validation-tweets']
    crossValidate(cvDb, maxCrossValidateIterations, featureExtractor, svms, typeMap)

    # run tests
    testDb = couch['test-tweets']
    test(testDb, maxTestIterations, featureExtractor, svms, typeMap, outputFileName)


def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-db',     help="name of the database which contains training data", type=str)
    parser.add_argument('-method', help="the machine learning method/algorithm to invoke. Currently supports: svm, naive-bayes",   type=str)
    parser.add_argument('-output', help="filename to which to write output of test data (will be filename.csv)", type=str)
    parser.add_argument('-training_iterations', help="The number of training iterations", type=int, default=maxTrainingIterations)
    args = parser.parse_args()
    classify(args.db, args.method, args.output, args.training_iterations);

if __name__ == '__main__':
    main()
