import fire, time, math, tqdm, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import Visualizer
from data.data import TextData
import models
from config import opt

from ipdb import set_trace


vis = Visualizer(opt.env)
def repackage_hidden(h):
	if type(h) == Variable:
		return Variable(h.data)
	else:
		return tuple(repackage_hidden(v) for v in h)

def main(**kwargs):
	opt.update(kwargs)
	vis.reinit(opt.env)
	#torch.cuda.manual_seed(args.seed)
	train_dataset = TextData(opt)
	train_dataLoader = DataLoader(
						train_dataset,
						batch_size = opt.batch_size,
						shuffle = opt.shuffle,
						num_workers = opt.num_workers,
						drop_last = True
					)
	valid_dataset = TextData(opt, is_train=False)
	valid_dataLoader = DataLoader(
						valid_dataset,
						batch_size = opt.batch_size,
						shuffle = opt.shuffle,
						num_workers = opt.num_workers,
						drop_last = True
					)
	model = getattr(models, opt.model)(opt).cuda()
	criterion = nn.CrossEntropyLoss()
	lr = opt.lr
	optimizer = optim.Adam(model.parameters(), lr=lr)
	best_val_loss = None
	
	for epoch in range(1, opt.epochs+1):
		print epoch
		#train
		epoch_start_time = time.time()
		model.train()
		hidden = model.init_hidden(opt.batch_size)
		total_loss = 0
		start_time = time.time()
		for i, batch in tqdm.tqdm(enumerate(train_dataLoader)):
			# set_trace()
			input = Variable(batch[0].cuda(), volatile=False)
			target = Variable(batch[1].cuda(), volatile=False)
			keyword = Variable(batch[2].cuda(), volatile=False)
			hidden = repackage_hidden(hidden)
			optimizer.zero_grad()
			model.zero_grad()
			output, hidden = model(input, hidden, keyword)
			loss = criterion(output.view(output.size(0)*output.size(1), output.size(2)), target.view(target.size(0)*target.size(1)))
			loss.backward()
			torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)
			optimizer.step()
			total_loss += loss.data
			if i % opt.log_interval == 0 and i > 0:
				cur_loss = total_loss[0] / opt.log_interval
				elapsed = time.time() - start_time
				vis.plot("loss", cur_loss)
				vis.plot("ppl", math.exp(cur_loss))
				#print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
				#'loss {:5.2f} | ppl {:8.2f}'.format(
				#	epoch, i, len(train_dataset) // opt.seq_len, lr,
				#	elapsed * 1000 / opt.log_interval, cur_loss, math.exp(cur_loss)))
				total_loss = 0
				start_time = time.time()
			if os.path.isfile("debug"):
				set_trace()
		#valid
		epoch_start_time = time.time()
		model.eval()
		total_loss = 0
		hidden = model.init_hidden(opt.batch_size)
		for i, batch in tqdm.tqdm(enumerate(valid_dataLoader)):
			input = Variable(batch[0].cuda(), volatile=False)#.cuda()
			target = Variable(batch[1].cuda())
			keyword = Variable(batch[2].cuda())
			output, hidden = model(input, hidden, keyword)
			loss = criterion(output.view(output.size(0)*output.size(1), output.size(2)), target.view(target.size(0)*target.size(1)))
			total_loss += loss.data
			hidden = repackage_hidden(hidden)
		val_loss = total_loss[0] / i
		vis.log('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
		if not best_val_loss or val_loss < best_val_loss:
			with open("%s/%s_%f.model" % (opt.checkpoint, opt.env, val_loss), 'wb') as f:
				torch.save(model, f)
			best_val_loss = val_loss

def test():
    # Run on test data.
	model.eval()
	total_loss = 0
	hidden = model.init_hidden(eval_batch_size)
	for batch in test_dataloader:
		data = Variable(source[i:i+seq_len], volatile=True)
		target = Variable(source[i+1:i+1+seq_len].view(-1))
		output, hidden = model(data, hidden)
		output_flat = output.view(-1, ntokens)
		total_loss += len(data) * criterion(output_flat, targets).data
		hidden = repackage_hidden(hidden)
	test_loss = total_loss[0] / len(v_dataloader)

	print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
		test_loss, math.exp(test_loss)))

if __name__ == "__main__":
	fire.Fire()
