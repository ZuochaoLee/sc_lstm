import os
import numpy as np
import torch as t
from torch.utils.data import Dataset

class TextData(Dataset):
	def __init__(self, opt, is_train=True):
		self.is_train = is_train
		self.seq_len = opt.seq_len
		text_train_file = "%s/text_train.npy" % opt.path
		text_valid_file = "%s/text_valid.npy" % opt.path
		keyword_train_file = "%s/keyword_train.npy" % opt.path
		keyword_valid_file = "%s/keyword_valid.npy" % opt.path
		if self.is_train:
			self.keyword = np.load(keyword_train_file)
			self.text = np.load(text_train_file)
		else:
			self.keyword = np.load(keyword_valid_file)
			self.text = np.load(text_valid_file)
		self.length = len(self.text)

	def __getitem__(self, index):
		if len(self.text[index]) >= self.seq_len + 1:
			data = self.text[index][:self.seq_len + 1]
		else:
			data = np.hstack((self.text[index], np.zeros(self.seq_len + 1 - len(self.text[index]), dtype=int)))
		return t.from_numpy(np.array(data[:self.seq_len])), t.from_numpy(np.array(data[1:])), t.from_numpy(np.array(self.keyword[index]))

	def __len__(self):
		return self.length
