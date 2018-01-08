import torch.nn as nn
from torch.autograd import Variable
from sc_lstm import SCLSTM

class CsLstmModel(nn.Module):
	def __init__(self, opt):
		super(CsLstmModel, self).__init__()
		self.ninp = opt.ninp
		self.nhid = opt.nhid
		self.nlayers = opt.nlayers
		self.ntoken = opt.ntoken
		self.embedding = nn.Embedding(opt.ntoken, opt.ninp) 
		self.drop = nn.Dropout(opt.dropout) 
		self.lstm = SCLSTM(opt.ninp, opt.nhid, opt.nlayers, opt.keyword_voc_size)
		self.fc = nn.Linear(opt.nhid, opt.ntoken)
		self.batch_size = opt.batch_size
		if opt.tie_weights:
			if opt.nhid != opt.ninp:
				raise ValueError("when using the tied flag, nhid must be equal to emsize")
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.embedding.weight.data.uniform_(-initrange, initrange)
		self.fc.bias.data.fill_(0)
		self.fc.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden, d_act):
		emb = self.drop(self.embedding(input))
		emb = emb.transpose(0, 1)
		output, hidden = self.lstm(emb, hidden, d_act)
		output = self.drop(output)
		output_ = self.fc(output.view(output.size(0) * output.size(1), output.size(2)))
		return output_.view(output.size(0), output.size(1), self.ntoken), hidden
		
	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		return (Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_()),
				Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_()))

