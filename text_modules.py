from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import pickle


#device = torch.device("cuda:0")



class TextEncoder(nn.Module):
	def __init__(self, batch_size, word_dim, embed_size, vocab_size, num_layers=2, use_abs=False, bidirectional = True, dropout = 0.0):
		super().__init__()
		self.emd = nn.Embedding(vocab_size, word_dim)
		self.gru = nn.GRU(word_dim, 1024, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)#.to(device);
		self.fc1_out = nn.Linear(1024, 2*embed_size)#.to(device)
		self.embed_size = embed_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.batch_size = batch_size
		self.relu = nn.ReLU()
		#self.init_weights()
		self.register_buffer('scale', torch.ones(1))
		self.register_buffer('shift', torch.zeros(1,embed_size))
		self.training = True
		self.scale_init = True
		self.max_length = 40

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		#self.emd.weight.data.uniform_(-0.1, 0.1)
		r = np.sqrt(6.) / np.sqrt(self.fc1.in_features +
								  self.fc1.out_features)
		self.fc1_out.weight.data.uniform_(-r, r)
		self.fc1_out.bias.data.fill_(0)

	def init_hidden(self,hidsize):
		h = Variable(torch.zeros(self.num_layers * (2 if self.bidirectional else 1), self.batch_size, hidsize), requires_grad=True).cuda()
		return h

	def forward_BiGRU(self, x, lengths):
		outs = []
		x = self.emd(x)
		hiddens = self.init_hidden(1024)
		emb = pack_padded_sequence(x, lengths, batch_first=True)
		self.gru.flatten_parameters()
		outputs, hidden_t = self.gru(emb, hiddens)
		output_unpack = pad_packed_sequence(outputs, batch_first=True)
		hidden_o = output_unpack[0].sum(1)
		hidden_o = hidden_o[:,:1024]+hidden_o[:,1024:]
		output_lengths = Variable(lengths.float().cuda(), requires_grad=False)
		hidden_o = torch.div(hidden_o, output_lengths.view(-1, 1))
		output = self.fc1_out(hidden_o)
		return output

	def forward(self, x, lengths):
		out = self.forward_BiGRU(x, lengths)
		return out

class TextDecoder(nn.Module):
	def __init__(self, batch_size, hidden_size, vocab_size, embed_size = 1024, num_layers=1, max_length=40, bidirectional = False):
		super().__init__()
		self.hidden_size = hidden_size
		self.embed_size = embed_size
		self.emd = nn.Embedding(vocab_size*2, hidden_size)#.to(device)
		self.gru = nn.LSTM(1536, 512, num_layers, bidirectional = bidirectional)#.to(device)
		self.out = nn.Linear(1*512, vocab_size)#.to(device)
		self.softmax = nn.LogSoftmax(dim=1)
		self.word_dropout_rate = 0.8
		self.embedding_dropout = nn.Dropout(p=0.5)
		self.vocab_size = vocab_size
		self.fc1 = nn.Linear(embed_size,hidden_size)#.to(device)
		self.bidirectional = bidirectional
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.max_length = max_length



	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""

		self.emd.weight.data.uniform_(-0.1, 0.1)
		r = np.sqrt(6.) / np.sqrt(self.fc1.in_features +
								  self.fc1.out_features)
		self.fc1.weight.data.uniform_(-r, r)
		self.fc1.bias.data.fill_(0)

	def init_hidden(self,):
		h = Variable(torch.zeros(self.num_layers * (2 if self.bidirectional else 1), self.batch_size, 512), requires_grad=True).cuda()
		c = Variable(torch.zeros(self.num_layers * (2 if self.bidirectional else 1), self.batch_size, 512), requires_grad=True).cuda()
		return (h,c)

	def train(self,input_sequence, hidden,lengths):
		hidden_t = self.init_hidden()
		hidden = hidden.unsqueeze(1)
		hidden = hidden.repeat(1,self.max_length,1)
		input_embedding = self.emd(input_sequence)
		if self.word_dropout_rate > 0:
			prob = torch.rand(input_sequence.size()).cuda()
			prob[(input_sequence.data - 1) * (input_sequence.data - 0) == 0] = 1
			decoder_input_sequence = input_sequence.clone()
			decoder_input_sequence[prob < self.word_dropout_rate] = 3
			input_embedding = self.emd(decoder_input_sequence)
			
		input_embedding = self.embedding_dropout(input_embedding)
		input_embedding = torch.cat([hidden,input_embedding],dim=2)
		packed_input = pack_padded_sequence(input_embedding, lengths, batch_first=True)

		# decoder forward pass
		outputs, _ = self.gru(packed_input,hidden_t)

		# process outputs
		padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0]
		padded_outputs = padded_outputs.contiguous()
		b,s,_ = padded_outputs.size()
		logp = nn.functional.log_softmax(self.out(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
		logp = logp.view(b, s, self.vocab_size)
		return logp

	def inference(self, hidden, max_length):
		dec_hidden = self.init_hidden()
		hidden = hidden.unsqueeze(1)
		input_sequence = torch.LongTensor(np.ones((self.batch_size,1)).astype(np.float64))#.to(device)
		lengths = torch.LongTensor(np.ones((self.batch_size,)).astype(np.int32))#.to(device)
		output_seq = []

		for _ in range(max_length):
			input_embedding = self.emd(input_sequence)
			input_embedding = torch.cat([hidden,input_embedding],dim=2)
			packed_input = pack_padded_sequence(input_embedding, lengths, batch_first=True)
			outputs, dec_hidden = self.gru(packed_input,dec_hidden)
			padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0]
			padded_outputs = padded_outputs.contiguous()
			out = self.out(padded_outputs).view(self.batch_size, self.vocab_size)
			out = torch.argmax(out,dim=1).unsqueeze(1)
			input_sequence = out

			output_seq.append(out.detach().clone().cpu().numpy())

		output_seq = np.concatenate(output_seq,axis=1)
		return output_seq

	def forward(self,input_sequence, hidden, lengths,train=True):
		if train:
			return self.train(input_sequence, hidden, lengths)
		else:
			return self.beam_search(hidden,lengths)




	def beam_search(self, hidden, seq_max_len=40, beam_width=2):
		words = Variable(torch.Tensor([1]).long(), requires_grad=False).repeat(self.batch_size).view(self.batch_size, 1, 1)
		probs = Variable(torch.zeros(self.batch_size, 1))  # [batch, beam]
		if torch.cuda.is_available():
			words = words.cuda()
			probs = probs.cuda()
		h = self.init_hidden()
		_hidden = hidden.unsqueeze(1)
		all_hidden = h[0].unsqueeze(3)  # [1, batch, lstm_dim, beam]
		all_cell = h[1].unsqueeze(3)
		all_words = words  # [batch, length, beam]
		all_probs = probs  # [batch, beam]
		for t in range(seq_max_len):
			new_words = []
			new_hidden = []
			new_cell = []
			new_probs = []
			tmp_words = all_words.split(1, 2)
			tmp_probs = all_probs.split(1, 1)
			tmp_hidden = all_hidden.split(1, 3)
			tmp_cell = all_cell.split(1, 3)
			for i in range(len(tmp_words)):
				last_word = tmp_words[i].split(1, 1)[-1].view(self.batch_size,1)
				inputs = self.emd(last_word)
				inputs = torch.cat([_hidden,inputs],dim=2).view(1, self.batch_size, -1).contiguous()
				last_state = (tmp_hidden[i].squeeze(3).contiguous(),tmp_cell[i].squeeze(3).contiguous())
				hidden, states = self.gru(inputs, last_state)
				probs = nn.functional.log_softmax((self.out(hidden)), dim=2) # [1, batch, vocab_size]
				probs, indices = probs.topk(beam_width, 2)
				probs = probs.view(self.batch_size, beam_width)  # [batch, beam]
				indices = indices.permute(1, 0, 2)  # [batch, 1, beam]
				tmp_words_rep = tmp_words[i].repeat(1, 1, beam_width)
				probs_cand = tmp_probs[i] + probs  # [batch, beam]
				words_cand = torch.cat([tmp_words_rep, indices], 1)  # [batch, length+1, beam]
				hidden_cand = states[0].unsqueeze(3).repeat(1, 1, 1, beam_width)  # [1, batch, lstm_dim, beam]
				cell_cand = states[1].unsqueeze(3).repeat(1, 1, 1, beam_width)
				new_words.append(words_cand)
				new_probs.append(probs_cand)
				new_hidden.append(hidden_cand)
				new_cell.append(cell_cand)
			new_words = torch.cat(new_words, 2)  # [batch, length+1, beam*beam]
			new_probs = torch.cat(new_probs, 1)  # [batch, beam*beam]
			new_hidden = torch.cat(new_hidden, 3)  # [1, batch, lstm_dim, beam*beam]
			new_cell = torch.cat(new_cell, 3) 
			probs, idx = new_probs.topk(beam_width, 1)  # [batch, beam]
			idx_words = idx.view(self.batch_size, 1, beam_width)
			idx_words = idx_words.repeat(1, t+2, 1)
			idx_states = idx.view(1, self.batch_size, 1, beam_width).repeat(self.num_layers, 1, 512, 1)
			# reduce the beam*beam candidates to top@beam candidates
			all_probs = probs
			all_words = new_words.gather(2, idx_words)
			all_hidden = new_hidden.gather(3, idx_states)
			all_cell = new_cell.gather(3, idx_states)
		#idx = all_probs.argmax(1)  # [batch]
		idx = idx.view(self.batch_size, beam_width,1, 1).repeat(1,1, seq_max_len+1, 1)
		captions = all_words.cpu().numpy() #.gather(2, idx)#.squeeze(2)  # [batch, length]
		return captions
