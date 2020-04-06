from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math
import scipy

device = torch.device("cuda:0")


''' Contains modules for flow layers for learning conditional priors '''

def split_feature_fc(x,type = 'split'):
	#return x[:,1::2], x [:,0::2]
	return x[:,0:int(x.size(1)/2)], x[:,int(x.size(1)/2):]

def cpd_sum(tensor, dim=None, keepdim=False):
	if dim is None:
		# sum up all dim
		return torch.sum(tensor)
	else:
		if isinstance(dim, int):
			dim = [dim]
		dim = sorted(dim)
		for d in dim:
			tensor = tensor.sum(dim=d, keepdim=True)
		if not keepdim:
			for i, d in enumerate(dim):
				tensor.squeeze_(d-i)
		return tensor


def cpd_mean(tensor, dim=None, keepdim=False):
	if dim is None:
		# mean all dim
		return torch.mean(tensor)
	else:
		if isinstance(dim, int):
			dim = [dim]
		dim = sorted(dim)
		for d in dim:
			tensor = tensor.mean(dim=d, keepdim=True)
		if not keepdim:
			for i, d in enumerate(dim):
				tensor.squeeze_(d-i)
		return tensor
		
		

class NN_net_fc(nn.Module):
	#split
	#shift
	#scale

	def __init__(self, in_channels, out_channels, hiddden_channels):
		super().__init__()
		self.fc1 = nn.Linear(in_channels, hiddden_channels)
		#self.fc2 = nn.Linear(hiddden_channels, hiddden_channels)
		self.fc3 = nn.Linear(hiddden_channels, out_channels)
	
	def forward(self,x):
		x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
		

class Affine_coupling_glow(nn.Module):
	'''Affine coupling layers as defined in [GLOW] '''
	def __init__(self, in_channels, out_channels, hiddden_channels,cond_dim):
		super(Affine_coupling_glow, self).__init__()
		if cond_dim is not None:
			self.NN_net = NN_net_fc(in_channels//2 + cond_dim, out_channels, hiddden_channels)
		else:
			self.NN_net = NN_net_fc(in_channels//2, out_channels, hiddden_channels)	

	def split(self,x,_type="split"):
		return split_feature_fc(x,_type)


	def forward_inference(self,x,logdet,cond):
		z1,z2  	= self.split(x)
		y2     	= self.NN_net(torch.cat((z1,cond), dim=1) if cond is not None else z1)
		shift, scale = self.split(y2,"cross")
		scale 		= torch.sigmoid(scale + 2.0)
		z2		= z2*scale
		z2	    = shift + z2
		logdet = cpd_sum(torch.log(scale), dim=[1]) + logdet
		z      = torch.cat((z1, z2), dim=1)
		return z, logdet

	def reverse_sampling(self,x,logdet,cond):
		z1,z2  	= self.split(x)
		y2     	= self.NN_net(torch.cat((z1,cond), dim=1) if cond is not None else z1)
		shift, scale 	= self.split(y2,"cross")
		scale 		= torch.sigmoid(scale + 2.0)
		z2	    = z2-shift
		z2		= z2/scale
		logdet = logdet - cpd_sum(torch.log(scale), dim=[1])
		z = torch.cat((z1, z2), dim=1)
		return z, logdet

	def forward(self, input, logdet = 0., cond = None, reverse=False):
		if not reverse:
			x, logdet = self.forward_inference(input, logdet, cond)
		else:
			x, logdet = self.reverse_sampling(input, logdet, cond)
		return x, logdet

class Affine_coupling(nn.Module):
	'''Affine coupling layers as defined in [NICE] '''
	def __init__(self, in_channels, out_channels, hiddden_channels, cond_dim):
		super(Affine_coupling,self).__init__()

		if cond_dim is not None:
			#print('cond_dim: ', cond_dim)
			in_dims = in_channels//2 + cond_dim
		else:
			in_dims = in_channels//2

		self.NN_net1 = NN_net_fc(in_dims, out_channels//2, hiddden_channels)
		self.NN_net2 = NN_net_fc(in_dims, out_channels//2, hiddden_channels)
		self.NN_net3 = NN_net_fc(in_dims, out_channels//2, hiddden_channels)
		self.NN_net4 = NN_net_fc(in_dims, out_channels//2, hiddden_channels)
		w_init = np.ones((1,in_channels))
		self.register_parameter("scale", nn.Parameter(torch.Tensor(w_init), requires_grad=True))
		
	def split(self,x,_type="split"):
		return split_feature_fc(x,_type)


	def forward_inference(self,x,logdet,cond):
		x1,x2  	= self.split(x)
		h11     =  x1
		h12     = x2+self.NN_net1(torch.cat((x1,cond), dim=1) if cond is not None else x1)
		h22     = h12 
		h21     =  h11 + self.NN_net2(torch.cat((h12,cond), dim=1) if cond is not None else h12)
		h31     = h21 
		h32     =  h22 + self.NN_net3(torch.cat((h21,cond), dim=1) if cond is not None else h21)
		h42     = h32 
		h41     =  h31 + self.NN_net4(torch.cat((h32,cond), dim=1) if cond is not None else h32)
		scale   = torch.sigmoid(self.scale + 5.0)
		h       =  scale*torch.cat((h41, h42), dim=1)
		logdet =  logdet + cpd_sum(torch.log(scale), dim=[1])
		return h, logdet

	def reverse_sampling(self,x,logdet,cond):
		scale = torch.sigmoid(self.scale + 5.0)
		h = x/scale
		#h = x/self.scale
		h41,h42 = self.split(h)

		h32  = h42
		h31 = h41 - self.NN_net4(torch.cat((h42,cond), dim=1) if cond is not None else h42)

		h21  = h31
		h22 = h32 - self.NN_net3(torch.cat((h31,cond), dim=1) if cond is not None else h31)

		h12  = h22 
		h11 = h21 - self.NN_net2(torch.cat((h22,cond), dim=1) if cond is not None else h22)

		x1  = h11 
		x2 = h12 - self.NN_net1(torch.cat((h11,cond), dim=1) if cond is not None else h11)

		logdet = logdet - cpd_sum(torch.log(scale), dim=[1])
		z = torch.cat((x1, x2), dim=1)
		return z, logdet

	def forward(self, input, logdet = 0., cond=None, reverse=False):
		if not reverse:
			x, logdet = self.forward_inference(input, logdet, cond)
		else:
			x, logdet = self.reverse_sampling(input, logdet, cond)
		return x, logdet	



class Switch(nn.Module):
	def __init__(self):
		super().__init__()

	def split(self,x,_type="split"):
		return split_feature_fc(x,_type)
		
	def forward(self, input, logdet=None, reverse=False):
		if not reverse:
			z1,z2  	= self.split(input)
			z = torch.cat((z2, z1), dim=1)
			return z, logdet
		else:
			z1,z2  	= self.split(input)
			z = torch.cat((z2, z1), dim=1)
			return z, logdet				

class Cond_ActNorm(nn.Module):
	def __init__(self, in_out_dim, hidden_channels, cond_dim):
		super().__init__()
		self.NN_net = NN_net_fc(cond_dim, 2*in_out_dim, hidden_channels).to(device)
		self.in_out_dim = in_out_dim

	def _center(self, input, reverse=False):
		if not reverse:
			return input + self.bias
		else:
			return input - self.bias

	def _scale(self, input, logdet=None, reverse=False):
		scale = torch.sigmoid(self.logs + 2.0)
		if not reverse:
			input = input * scale
		else:
			input = input / scale
		if logdet is not None:
		
			dlogdet = torch.sum(torch.log(scale),dim=1) 
			if reverse:
				dlogdet *= -1
			logdet = logdet + dlogdet
		return input, logdet

	def forward(self, input, logdet=0, cond=None, reverse=False):
		nno = self.NN_net(cond)
		self.bias = nno[:,:self.in_out_dim]
		self.logs = nno[:,self.in_out_dim:]
		if not reverse:
			# center and scale
			input = self._center(input, reverse)
			input, logdet = self._scale(input, logdet, reverse)
		else:
			# scale and center
			input, logdet = self._scale(input, logdet, reverse)
			input = self._center(input, reverse)
		return input, logdet



class InvertibleConv1x1(nn.Module):
	def __init__(self, num_channels, LU_decomposed=True):
		super().__init__()
		w_shape = [num_channels, num_channels]
		w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
		if not LU_decomposed:
			# Sample a random orthogonal matrix:
			self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
		else:
			np_p, np_l, np_u = scipy.linalg.lu(w_init)
			np_s = np.diag(np_u)
			np_sign_s = np.sign(np_s)
			np_log_s = np.log(np.abs(np_s))
			np_u = np.triu(np_u, k=1)
			l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
			eye = np.eye(*w_shape, dtype=np.float32)

			self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
			self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
			self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
			self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
			self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
			self.l_mask = torch.Tensor(l_mask)
			self.eye = torch.Tensor(eye)
		self.w_shape = w_shape
		self.LU = LU_decomposed

	def get_weight(self, input, reverse):
		w_shape = self.w_shape
		pixels = list(input.size())[-1]
		if not self.LU:

			#thops.pixels(input)
			dlogdet = (torch.slogdet(self.weight)[1]) #* pixels*pixels
			if not reverse:
				weight = self.weight#.view(w_shape[0], w_shape[1], 1, 1)
			else:
				weight = torch.inverse(self.weight.double()).float()#.view(w_shape[0], w_shape[1], 1, 1)
			return weight, dlogdet
		else:
			self.p = self.p.to(input.device)
			self.sign_s = self.sign_s.to(input.device)
			self.l_mask = self.l_mask.to(input.device)
			self.eye = self.eye.to(input.device)
			l = self.l * self.l_mask + self.eye
			u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
			dlogdet = cpd_sum(self.log_s) #* pixels*pixels
			if not reverse:
				w = torch.matmul(self.p, torch.matmul(l, u))
			else:
				l = torch.inverse(l.double().cpu()).float()
				u = torch.inverse(u.double().cpu()).float()
				w = torch.matmul(u, torch.matmul(l, self.p.cpu().inverse())).cuda()
			return w, dlogdet #view(w_shape[0], w_shape[1], 1, 1)

	def forward(self, input, logdet=None, reverse=False):
		weight, dlogdet = self.get_weight(input, reverse)
		if not reverse:
			z = torch.mm(input, weight)
			if logdet is not None:
				logdet = logdet + dlogdet
			return z, logdet
		else:
			z = torch.mm(input, weight)
			if logdet is not None:
				logdet = logdet - dlogdet
			return z, logdet


class FlowStep_fc(nn.Module):
	def __init__(self,in_channels, out_channels, hidden_channels,cond_dim,coupling):
		super().__init__()
		if coupling == 'linear':
			self.affine_coupling = Affine_coupling(in_channels, out_channels, hidden_channels,cond_dim)#.to(device)
			self.switch = Switch()
		elif coupling == 'full':
			self.actnorm = Cond_ActNorm(in_channels, hidden_channels,cond_dim) 
			self.affine_coupling = Affine_coupling_glow(in_channels, out_channels, hidden_channels,cond_dim)#.to(device)
			self.invert_1x1_layer = InvertibleConv1x1(in_channels)
			self.switch = Switch()
		else:
			self.affine_coupling = Affine_coupling(in_channels, out_channels, hidden_channels,cond_dim)	
			self.switch = Switch()

		self.coupling = coupling	


	def forward_inference(self,x,logdet=0., cond=None, reverse=False):
		if self.coupling == 'full':
			x, logdet = self.actnorm(x, logdet, cond, reverse)
			#x, logdet = self.invert_1x1_layer(x, logdet,reverse)
		x, logdet = self.affine_coupling(x, logdet, cond, reverse)
		x, logdet = self.switch(x, logdet, reverse)
		return x, logdet

	def reverse_sampling(self,x,logdet=0., cond=None, reverse=True):
		x, logdet = self.switch(x, logdet, reverse)
		x, logdet = self.affine_coupling(x, logdet, cond, reverse)
		if self.coupling == 'full':
			#x, logdet = self.invert_1x1_layer(x, logdet,reverse)
			x, logdet = self.actnorm(x, logdet, cond, reverse)
		return x, logdet

	def forward(self,input,logdet=0., cond=None, reverse=False):
		if not reverse:
			z, logdet = self.forward_inference(input,logdet,cond,reverse)
		else:
			z, logdet = self.reverse_sampling(input,logdet,cond,reverse)

		return z, logdet


class FlowNet_fc(nn.Module):
	def __init__(self, input_dim, hidden_channels, K, L, cond_dim=2,coupling='linear'):
		super().__init__()
		self.layers = nn.ModuleList()
		self.K = K
		self.L = L
		for i in range(L):
			for _ in range(K):
				self.layers.append(
					FlowStep_fc(in_channels=input_dim,out_channels=input_dim, 
						hidden_channels=hidden_channels,cond_dim=cond_dim,coupling=coupling))
				

	def forward(self, input, logdet=0., cond=None, reverse=False, eps_std=None):
		if not reverse:
			return self.encode(input, logdet, cond)
		else:
			return self.decode(input, cond, eps_std)

	def encode(self, z, logdet=0.0, cond=None):
		for layer in self.layers:
			z, logdet = layer(z, logdet, cond=cond, reverse=False)
		return z, logdet

	def decode(self, z, cond, eps_std=None):
		for layer in reversed(self.layers):
			z, logdet = layer(z, logdet=0, cond=cond, reverse=True)
		return z




class FlowLatent(nn.Module):

	def __init__(self,batch_size,input_dim,hidden_channels=2048,K=16,gaussian_dims=(1024-700),gaussian_var=0.1,cond_dim=None,coupling='linear'):
		super().__init__()
		self.flow = FlowNet_fc(input_dim= input_dim,
							hidden_channels=hidden_channels,
							K=K,
							L=1,
							cond_dim=cond_dim,
							coupling=coupling
							)#.to(device)

		self.input_dim = input_dim
		self.relu = nn.ReLU()
		self.mean = torch.zeros(batch_size,self.input_dim).float().to(device)
		self.logs = torch.zeros(batch_size,self.input_dim).float().to(device)
		if gaussian_var != 0:
			self.logs[:,-gaussian_dims:] = torch.ones(batch_size,gaussian_dims).float()*np.log(gaussian_var)
		
		self.gaussian_dims = gaussian_dims


	def forward(self, x=None, z_im=None, z=None, cond=None, eps_std=None, reverse=False, tstep = -1):			
		if not reverse:
			#code for forward pass in normalizing flow
			return self.normal_flow(x, z_im, cond=cond)
		else:
			#code for reverse pass in normalizing flow
			return self.reverse_flow(x, z, cond=cond, eps_std=eps_std)

	def normal_flow(self, x, z_im, cond=None):
		x_shape = list(x.size())#[-1]
		pixels = x_shape[1]
		z = x 
		logdet = torch.zeros_like(x[:, 0])
		logdet = logdet
		# encode
		z, objective = self.flow(z, logdet=logdet, cond=cond, reverse=False)
		
		if self.gaussian_dims > 0:
			objective = objective.to(device) + GaussianDiag.logp(self.mean[:,-self.gaussian_dims:], self.logs[:,-self.gaussian_dims:], z[:,-self.gaussian_dims:].to(device))
		nll = (-objective.to(device)) / float(np.log(2.)*x_shape[1])
		return z, nll, None

	def reverse_flow(self, _x, z, cond=None, eps_std=None):
		if z is None:
			if _x is not None: 
				if self.gaussian_dims != 0: 
					z = GaussianDiag.sample(self.mean, self.logs, eps_std)
					z = torch.cat((_x,z[:,-self.gaussian_dims:]), dim=1)
				else:
					z = _x	
			else:
				z = GaussianDiag.sample(self.mean, self.logs, eps_std)		
		else:
			if _x is not None: 
				z = torch.cat((_x,z[:,-self.gaussian_dims:]), dim=1)
			else:
				z = GaussianDiag.sample(self.mean, self.logs, eps_std)		

		x = self.flow(z, cond=cond, eps_std=eps_std, reverse=True)
		return x, z


class GaussianDiag:
	Log2PI = float(np.log(2 * np.pi))

	@staticmethod
	def likelihood(mean, logs, x):
		"""
		lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
			  k = 1 (Independent)
			  Var = logs ** 2
		"""
		return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI) + torch.exp(logs * 2.)

	@staticmethod
	def logp(mean, logs, x):
		likelihood = GaussianDiag.likelihood(mean, logs, x)
		return cpd_sum(likelihood, dim=1)

	@staticmethod
	def sample(mean, logs, eps_std=None):
		eps_std = eps_std or 1
		eps = torch.normal(mean=torch.zeros_like(mean),
						   std=torch.ones_like(logs) * eps_std)
		return mean + torch.exp(logs) * eps