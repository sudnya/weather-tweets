# Author: Sudnya Padalikar
# Date  : Dec 23 2013
# Brief : A python script to read the csv data from Kaggle dataset into the couch-db database
# Comment : I will try to abide by https://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Python_Style_Rules

#!/usr/bin/python
import csv
import sys
import argparse
import couchdb

def cleanUp(tweet):
    newTweet = {}
    for key, value in tweet.iteritems():
        newTweet[unicode(key, errors='ignore').encode('ascii', 'ignore')] = unicode(value, errors='ignore').encode('ascii', 'ignore')

    return newTweet


def csvToCouchdb(csvName, couchdbName, delPrev):
    # read csv into a dictionary
    csvDict = csv.DictReader(open(csvName, 'r'), delimiter=',', quotechar='"')
    dataList = [entry for entry in csvDict]
    # get server
    server = couchdb.Server()

    if couchdbName not in server:
        db = server.create(couchdbName)
    else:
        if (delPrev):
            del server[couchdbName]
            db = server.create(couchdbName)
        else:
            db = server[couchdbName]

    for entry in dataList:
        tweet = entry
        print dataList.index(entry), cleanUp(tweet)
        db[str(dataList.index(entry))] = cleanUp(tweet)


def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-csv',    help="path to the input csv file", type=str)
    parser.add_argument('-dbname', help="name of database to be created", type=str)
    parser.add_argument('-delete', help="delete db if it already exists", type=bool)
    args = parser.parse_args()
    csvToCouchdb(args.csv, args.dbname, args.delete)

if __name__ == '__main__':
    main()
