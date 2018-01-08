#
class option(object):
	def __init__(self):
		self.env = "sc_lstm"
		self.batch_size = 64
		self.num_workers = 4
		self.shuffle = True
		self.path = 'data/cbt'
		#model
		self.model = "CsLstm"
		self.ntoken = 51815
		self.keyword_voc_size = 16040
		self.ninp = 128
		self.nhid = 128
		self.nlayers = 2
		self.dropout = 0.2
		self.tie_weights = False
		#train
		self.lr = 0.01
		self.momentum = 0.9
		self.clip = 0.25
		self.epochs = 40
		self.seq_len = 100
		self.checkpoint = 'checkpoint/cbt'
		self.log_interval = 100
		#gen
		self.outf = "output.txt"
		self.words = 1000
		self.temperature = 1.0
		self.save = "best.pt"

	def update(self, kwargs):
		# print kwargs
		for k, v in kwargs.iteritems():
			if not hasattr(self, k):
				raise Exception("opt has not attribute <%s>" % k)
			setattr(self, k, v)

opt = option()
