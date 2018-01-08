import os, json, fire
import numpy as np

class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)

	def loadData(self, path):
		assert os.path.exists(path)
		self.add_word("<PAD>")
		self.add_word("<UKN>")
		self.add_word("<ESO>")
		with open(path, 'r') as f:
			for line in f:
				words = line.decode("utf8").split(" ") + ["<ESO>"]
				for word in words:
					self.add_word(word)
	
	def save(self, path):
		data = {}
		data["word2idx"] = self.word2idx		
		data["idx2word"] = self.idx2word
		with open(path, "w+") as f:
			f.write(json.dumps(data))
	
	def load(self, path):
		with open(path, "r") as f:
			text = f.read()
			data = json.loads(text)
			self.word2idx = data["word2idx"]
			self.idx2word = data["idx2word"]

def Tokenize(path):
	d = Dictionary()
	d.load("%s/dict.json" % path)
	res = [[], []]
	with open("%s/text_train.txt" % path, 'r') as tf, open("%s/text_valid.txt" % path) as vf:
		for i, f in enumerate([tf, vf]):
			for line in f:
				words = line.decode("utf8").split(" ")
				t_ = []
				for word in words:
					t_.append(d.word2idx[word])
				res[i].append(t_)			
	t_res = np.array(res[0])
	v_res = np.array(res[1])
	np.save("%s/text_train.npy" % path, t_res)
	np.save("%s/text_valid.npy" % path, v_res)
	print t_res.shape, v_res.shape

def CreKeyVec(path):
	d = Dictionary()
	d.loadData("%s/keyword_train.txt" % path)
	d.loadData("%s/keyword_valid.txt" % path)
	d.save("%s/keyword_dict.json" % path)
	exit()
	ntoken = len(d)
	ids = [[], []]
	with open("%s/keyword_train.txt" % path, "r") as tf, open("%s/keyword_valid.txt" % path, "r") as vf:
		for i, f in enumerate([tf, vf]):
			for line in f:
				words = line.decode("utf8").split(" ")
				ids_ = np.zeros(ntoken)
				for word in words:
					ids_[d.word2idx[word]] = 1
				ids[i].append(ids_)
	t_ids = np.array(ids[0])
	v_ids = np.array(ids[1])
	np.save("%s/keyword_train.npy" % path, t_ids)
	np.save("%s/keyword_valid.npy" % path, v_ids)
	print t_ids.shape, v_ids.shape

def CreDict(path):
	d = Dictionary()
	d.loadData("%s/text_train.txt" % path)
	d.loadData("%s/text_valid.txt" % path)
	d.save("%s/dict.json" % path)
	print "Create dictionary finished!! the length of dictionary is: %d" % len(d)

if __name__ == "__main__":
	fire.Fire()
