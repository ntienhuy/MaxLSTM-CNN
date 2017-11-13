import cPickle
import numpy as np
import math
import os
from gensim import models


def load_idsent(data):
	sent1 = []
	sent2 = []
	for row in data:
		sent1.append(row[7])
		sent2.append(row[8])
	maxlen = 0
	
	for sent in sent1:
		if len(sent) > maxlen:
			maxlen = len(sent)
	for sent in sent2:
		if len(sent) > maxlen:
			maxlen = len(sent)
	
	sent1 = np.array(sent1)
	sent2 = np.array(sent2)
	print ("Max len:",maxlen)
	return sent1,sent2
	
def load_data_idsent():
	my_path = os.path.abspath(os.path.dirname(__file__))
	train_data, test_data, dev_data = cPickle.load(open(my_path+"/corpus.p", "rb"))
	a,b = load_idsent(train_data)
	c,d = load_idsent(test_data)
	e,f = load_idsent(dev_data)
	return a,b,c,d,e,f

	
	

def load_target_distribution(data):
	scores = []
	for row in data:
		y = float(row[4])+1
		p = np.zeros(6)
		for i in range(1,7):
			if i == (math.floor(y) + 1):
				p[i-1] = y - math.floor(y)
			elif i == math.floor(y) :
				p[i-1] = math.floor(y) - y  + 1
			else:
				p[i-1] = 0
		scores.append(p)
	scores = np.array(scores)
	return scores
	
def load_data_target_distribution():
	my_path = os.path.abspath(os.path.dirname(__file__))
	train_data, test_data, dev_data = cPickle.load(open(my_path+"/corpus.p", "rb"))
	a = load_target_distribution(train_data)
	b = load_target_distribution(test_data)
	c = load_target_distribution(dev_data)
	return a,b,c
	
def load_score(data):
	scores = []
	for row in data:
		y = float(row[4])
		scores.append(y)
	scores = np.array(scores)
	return scores
	
def load_data_score():
	my_path = os.path.abspath(os.path.dirname(__file__))
	train_data, test_data, dev_data = cPickle.load(open(my_path+"/corpus.p", "rb"))
	a = load_score(train_data)
	b = load_score(test_data)
	c = load_score(dev_data)
	return a,b,c
	
