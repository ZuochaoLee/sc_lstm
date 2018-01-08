import fire, json
import torch
from torch.autograd import Variable

import models
from config import opt
from data.data import TextData

def main(keywords, **kwargs):
	opt.update(kwargs)
	
	print "loading model ..."
	with open("%s/%s" % (opt.checkpoint, opt.save), 'rb') as f:
		model = torch.load(f)
	model.eval()
	model.cuda()
	print "model format:"
	print model

	print "loading dict ..."
	with open("%s/dict.json" % opt.path) as f:
		dicts = json.loads(f.read())
	idx2word = dicts["idx2word"]
	word2idx = dicts["word2idx"]
	print "the dictionary lenght is %d" % len(idx2word)
	with open("%s/keyword_dict.json" % opt.path) as f:
		dicts = json.loads(f.read())
	key_idx2word = dicts["idx2word"]
	key_word2idx = dicts["word2idx"]
	print "the keyword dictionary lenght is %d" % len(key_idx2word)
	
	print "Keywords: %s" % keywords
	keywords = keywords.split(" ")
	keywords_ = torch.zeros(len(key_idx2word)).long()
	for keyword in keywords:
		keywords_[(word2idx[keyword])] = 1
	keywords = keywords_.squeeze(0)
	
	hidden = model.init_hidden(1)
	keywords = Variable(keywords)
	input = Variable(torch.rand(1, 1).mul(opt.ntoken).long(), volatile=True)
	input.data = input.data.cuda()
	keywords.data = keywords.data.cuda()
	print "the random first word is '%s'" % idx2word[input.data.squeeze().cpu().numpy()[0]]
	with open(opt.outf, 'w') as outf:
		for i in range(opt.words):
			output, hidden = model(input, hidden, keywords)
			word_weights = output.squeeze().data.div(opt.temperature).exp().cpu()
			word_idx = torch.multinomial(word_weights, 1)[0]
			input.data.fill_(word_idx)
			word = idx2word[word_idx]
			print word
			if word == "<ESO>":
				break
			outf.write(word.encode("utf-8") + ('\n' if i % 20 == 19 else ' '))
			if i % opt.log_interval == 0:
				print('| Generated {}/{} words'.format(i, opt.words))

if __name__ == "__main__":
	fire.Fire()
