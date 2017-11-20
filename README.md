# MaxLSTM-CNN
MaxLSTM-CNN employs multiple word embeddings for semantic textual similarity. The detail of this model can be found in [this paper](#)


Requirements
-------------
* NLTK (http://www.nltk.org/)
* Keras with Tensorflow backend (https://keras.io/)
* Gensim (https://radimrehurek.com/gensim/)

Pre-trained word embeddings
-------------
Put these following files into **wordEmbedding** folder:
* word2vec ([GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing))
* fastText ([wiki.en.vec](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec))
* Glove ([glove.840B.300d.txt](http://nlp.stanford.edu/data/glove.840B.300d.zip))
* Baroni ([EN-wform.w.5.cbow.neg10.400.subsmpl.txt](http://clic.cimec.unitn.it/composes/materials/EN-wform.w.5.cbow.neg10.400.subsmpl.txt.gz))
* SL999 ([paragram_300_sl999.txt](https://drive.google.com/file/d/0B9w48e1rj-MOck1fRGxaZW1LU2M/view?usp=sharing))




Training
-------------
Run:
```
python train.py
```
The performance should get about 82.2% in Pearson's score on [STS Benchmark](http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz) dataset.

