import csv
import nltk
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
np.random.seed(1336)  # for reproducibility
import cPickle


index = 1
word2id = {}
id2word = {}


def convert2id(word):
	global word2id
	global id2word
	global index
	word = word.lower()
	if not(word in word2id):
		word2id[word] = index
		id2word[index] = word
		index += 1
		
	return word2id[word]

def sent2id(sent):
	id_sequence = []
	tokens = nltk.word_tokenize(sent)	
	for word in tokens:
		id_sequence.append(convert2id(word))
	return id_sequence
	
	
i = 1	
def loadcsv(path):
	global i
	data = []	
	with open(path) as f:
		for line in f:
			if i%100 == 0:
				print i
			row = line.split('\t')			
			sentid1 = sent2id(row[5])
			sentid2 = sent2id(row[6])		
			new_row = row[0:7]
			new_row.append(sentid1)
			new_row.append(sentid2)
			data.append(new_row)
			i +=1
	return data




def readBaroni(path):
	model_org = {}
	with open(path) as f:
		for line in f:
			row = line.split(' ')
			model_org[row[0]] = np.array(row[1:])
	return model_org

def readBaroniO(path):
	model_org = {}
	with open(path) as f:
		for line in f:
			row = line.split('\t')
			model_org[row[0]] = np.array(row[1:])
	return model_org

def saveW(word2vec_path,bin,name,size):
	global word2id
	global id2word
	if word2vec_path == "wordEmbedding/EN-wform.w.5.cbow.neg10.400.subsmpl.txt":
		model_org = readBaroniO(word2vec_path)
	elif word2vec_path == "wordEmbedding/glove.840B.300d.txt":
		model_org = readBaroni(word2vec_path)
	elif word2vec_path == "wordEmbedding/paragram_300_sl999.txt":
		model_org = readBaroni(word2vec_path)
	else:
		model_org = KeyedVectors.load_word2vec_format(word2vec_path, binary=bin)
			
	#get W weight for embedding layer
	W = np.zeros(shape=(len(word2id)+1+2, size), dtype='float32')
	W[0] = np.zeros(size, dtype='float32')

	count_in = 0
	count_out = 0
	for word in word2id:
		i = word2id[word]
		if word in model_org:
			W[i] = model_org[word]
			count_in += 1
		else:
			W[i] = np.random.uniform(-0.25,0.25,size)
			count_out += 1

	print (name,"In-vocalulary size:",count_in,"Out-vocalulary size", count_out)
	cPickle.dump([W,word2id,id2word], open(name, "wb"))		
	
	
def run():	
	train_data = loadcsv('data/sts-train.csv')
	test_data = loadcsv('data/sts-test.csv')
	dev_data = loadcsv('data/sts-dev.csv')
	print ("Train size:",len(train_data),"Test size:", len(test_data),"Dev size:", len(dev_data))
	cPickle.dump([train_data,test_data,dev_data], open("corpus.p", "wb"))
	
	saveW('wordEmbedding/GoogleNews-vectors-negative300.bin', True,"word2vec_dic.p",300)
	saveW('wordEmbedding/wiki.en.vec', False,"fasttext_dic.p",300)
	saveW('wordEmbedding/paragram_300_sl999.txt', False,"sl999_dic.p",300)	
	saveW('wordEmbedding/glove.840B.300d.txt', False,"glove_dic.p",300)	
	saveW('wordEmbedding/EN-wform.w.5.cbow.neg10.400.subsmpl.txt', False,"baroni_dic.p",400)	