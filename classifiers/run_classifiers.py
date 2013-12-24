# Author: Sudnya Padalikar
# Date  : Dec 23 2013
# Brief : A python script to call various classifiers implemented in scikit learn
# Comment : I will try to abide by https://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Python_Style_Rules

#!/usr/bin/python
import sys
import couchquery
import argparse
from sklearn import svm

def svmClassify(dbname):
	print 'Opening couchdb named ' + dbname
	#db = Database('http://localhost:5984/'+dbname)


def main():
	parser = argparse.ArgumentParser(description="Process commandline inputs")
	parser.add_argument('-db',     help="name of the database which contains training data", type=str)
	parser.add_argument('-method', help="the machine learning method/algorithm to invoke",   type=str)
	args = parser.parse_args()
	if (args.method == "svm"):
		svmClassify(args.db);

if __name__ == '__main__':
    main()
