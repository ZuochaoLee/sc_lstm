import torch.nn as nn
import torch
from torch.autograd import Variable

class SeqCrossEntropyLoss(nn.Module):
	def __init__(self):
		super(SeqCrossEntropyLoss, self).__init__()
		self.CELoss = nn.CrossEntropyLoss

	def forward(self, output, target):
		lossList = []
		for t in range(output.size(o)):
			loss.append(self.CELoss(output[t,...], target[t,...]))
		
