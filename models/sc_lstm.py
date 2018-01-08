import torch.nn as nn
from torch.autograd import Variable
import torch

class SCLSTM_cell(nn.Module):
	"""Initialize a base cell of SC_LSTM
	Args:
      input_size: int that is the embedding size 
      hidden_size: int that is the hidden size
      keyword_voc_size: int that is the size of keywords voc
	"""
	def __init__(self, input_size, hidden_size, keyword_voc_size):
		super(SCLSTM_cell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.keyword_voc_size = keyword_voc_size
		self.dLinear = nn.Linear(keyword_voc_size, hidden_size)
		self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

	def forward(self, input, hidden_state, d_act):
		"""
        args:
            hidden_state: tuple (hidden, c)
            input: is the tensor of shape Batch, input_size
			d_act: is the tensor of shape Batch, keyword_voc_size
		"""
		hidden, c = hidden_state
		combined = torch.cat((input, hidden), 1)
		A = self.linear(combined)
		ai, af, ao, ag = torch.split(A, self.hidden_size, dim=1)
		i = torch.sigmoid(ai)
		f = torch.sigmoid(af)
		o = torch.sigmoid(ao)
		g = torch.tanh(ag)
		d = self.dLinear(d_act.type(torch.FloatTensor).cuda())        
		next_c = f * c + i * g + torch.tanh(d)
		next_h = o * torch.tanh(next_c)
		return next_h, next_c

	def init_hidden(self,batch_size):
		return (
			Variable(
				torch.zeros(batch_size, self.hidden_size)
			).cuda(),
			Variable(
				torch.zeros(batch_size,	self.hidden_size)
			).cuda()
		)

class DA_Cell(nn.Module):
	"""Initialize a base cell of DA
	Args:
      input_size: int that is the embedding size 
      hidden_size: int that is the hidden size
      keyword_voc_size: int that is the size of keywords voc
	"""
	def __init__(self, input_size, hidden_size, keyword_voc_size):
		super(DA_Cell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.keyword_voc_size = keyword_voc_size
		self.alpha = 0.5
		self.wLinear = nn.Linear(input_size, keyword_voc_size)
		self.hLinear = nn.Linear(hidden_size, keyword_voc_size)

	def forward(self, input, hidden, sc_vec):
		"""
        args:
            hidden: is the tenmsor of shape Batch, Hidden_size
            input: is the tensor of shape Batch, input_size
			d_act: is the tensor of shape Batch, keyword_voc_size
		"""
		wr = self.wLinear(input)
		hr = self.hLinear(hidden)
		rt = torch.sigmoid(wr + self.alpha * hr)
		sc_vec = sc_vec.type(torch.DoubleTensor) * rt.type(torch.DoubleTensor)
		return sc_vec

class SCLSTM(nn.Module):
	"""Initialize a Mutl-layers SC_LSTM
    Args:
      input_size: int that is the embedding size 
      hidden_size: int that is the hidden size
      num_layers: int that the num of SC_LSTM layers
      keyword_voc_size: int that is the size of keywords voc
	"""
	def __init__(self, input_size, hidden_size, num_layers, keyword_voc_size):
		super(SCLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.keyword_voc_size = keyword_voc_size
		self.num_layers = num_layers
		cell_list = []
		da_list = []
		cell_list.append(SCLSTM_cell(self.input_size, self.hidden_size, self.keyword_voc_size).cuda())
		da_list.append(DA_Cell(self.input_size, self.hidden_size, self.keyword_voc_size).cuda())
		for idcell in xrange(1,self.num_layers):
			cell_list.append(SCLSTM_cell(self.hidden_size, self.hidden_size, self.keyword_voc_size).cuda())
			da_list.append(DA_Cell(self.hidden_size, self.hidden_size, self.keyword_voc_size).cuda())
		self.cell_list = nn.ModuleList(cell_list)
		self.da_list = nn.ModuleList(da_list)

	def forward(self, input, hidden_state, d_act):
		"""
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len, Batch, input_size
			d_act is the tensor of shape Batch, keyword_voc_size
		"""
		next_hidden=[]
		seq_len=input.size(0)
		current_input = input

		for idlayer in xrange(self.num_layers):#loop for every layer
			hidden_c=hidden_state[idlayer]
			all_output = []
			output_inner = []
			for t in xrange(seq_len):#loop for every stepa
				d_act = self.da_list[idlayer](current_input[t,...], hidden_c[0], d_act)
				hidden_c=self.cell_list[idlayer](current_input[t,...], hidden_c, d_act)
				output_inner.append(hidden_c[0])
			next_hidden.append(hidden_c)
			current_input = torch.cat(output_inner, 0).view(input.size(0), input.size(1), self.hidden_size)
		return current_input, next_hidden

	def init_hidden(self,batch_size):
		init_states=[]
		for i in xrange(self.num_layers):
			init_states.append(self.cell_list[i].init_hidden(batch_size))
		return init_states

if __name__ == "__main__":
	input_size = 6
	hidden_size = 5
	batch_size = 4
	keyword_voc_size = 3
	nlayers=2
	seq_len=7
	input = Variable(torch.rand(seq_len, batch_size, input_size)).cuda()
	d_act = Variable(torch.rand(batch_size, keyword_voc_size)).cuda()
	sc_lstm=SCLSTM(input_size, hidden_size, nlayers, keyword_voc_size)
	sc_lstm.cuda()

	print 'sc_lstm module:', sc_lstm
	print 'params:'
	params=sc_lstm.parameters()
	for p in params:
		print 'param ', p.size()
		print 'mean ', torch.mean(p)

	hidden_state=sc_lstm.init_hidden(batch_size)
	print 'hidden_h shape ',len(hidden_state)
	print 'hidden_h shape ',hidden_state[0][0].size()
	out=sc_lstm(input, hidden_state, d_act)
	print 'out shape', out[1].size()
	print 'len hidden ', len(out[0])
	print 'next hidden', out[0][0][0].size()
	print 'sclstm dict', sc_lstm.state_dict().keys()

	L=torch.sum(out[1])
	L.backward()
