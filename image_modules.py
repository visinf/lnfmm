from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models


''' Contains modules for image encoders and image decoders'''

class VGG16Encoder(nn.Module):
	def __init__(self, encoded_dims, hidden_dims=1024):
		super(VGG16Encoder, self).__init__()
		self.vgg16 = models.vgg16(pretrained=True)
		fc_modules=list(self.vgg16.classifier.children())[:-1]
		self.vgg16.classifier=nn.Sequential(*fc_modules)
		'''for p in self.vgg16.parameters():
			p.requires_grad = False'''

		self.encoder = nn.Sequential(nn.Linear(4096,hidden_dims),
			nn.ReLU(),
			nn.Linear(hidden_dims,hidden_dims),
			nn.ReLU(),
			#nn.Linear(hidden_dims,hidden_dims).cuda(),nn.ReLU().cuda(),
			nn.Linear(hidden_dims,encoded_dims))

	def encode(self, x):
			x = self.vgg16(x)
			x = self.encoder(x)
			return x

	def forward(self, x):
			return self.encode(x)

''' Image decoders are based on generator network for SaGAN'''

class ImageEncoder(nn.Module):
	def __init__(self, img_dim, z_dim):
		super(ImageEncoder, self).__init__()

		self.conv_block = nn.Sequential(
			nn.Conv2d( 3, 128, 4, 2, 1),#128
			nn.BatchNorm2d(128),
			nn.ReLU(),
			#nn.Conv2d( 128, 128, 4, 2, 1),
			#nn.BatchNorm2d(128),
			#nn.ReLU(),
			nn.Conv2d( 128, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d( 128, 256, 4, 2, 1),#64
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d( 256, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d( 256, 512, 4, 2, 1),#32
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d( 512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d( 512, 512, 4, 2, 1),#16
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d( 512, 512, 4, 2, 1),#8
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d( 512, 128, 4, 2, 1),#4
			nn.BatchNorm2d(128),
			nn.ReLU())

		self.relu = nn.ReLU()
		
		self.fc1 = nn.Linear( 128*4*4, 2048)
		self.bn_f1 = nn.BatchNorm1d(2048);

		self.fc_out_mu = nn.Linear( 2048, img_dim)

		#self.fc_out_var = nn.Linear( 1024, z_dim)

		self.img_dim = img_dim
		self.z_dim = z_dim

	def encode(self, x): # Q(z|x)

		h4 = self.conv_block(x)

		h4 = h4.view(x.size(0),-1)

		h5 = self.relu(self.bn_f1(self.fc1(h4)))

		total_mu = self.fc_out_mu(h5)

		#logvar = self.fc_out_var(h5)

		#mu = total_mu[:,-self.z_dim:]

		#z_im = total_mu[:,:-self.z_dim]

		return total_mu#z_im, mu, logvar

	def reparametrize(self, mu, logvar):
		eps = torch.randn(logvar.size(0),logvar.size(1), device=device) #Variable(std.data.new(std.size()).normal_())
		return eps*torch.exp(logvar / 2) + mu

	def kl_loss(self, mu, logvar):
		return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)	

	def forward(self, input):
		z_im = self.encode(input) #, mu, logvar
		#z = self.reparametrize(mu, logvar)
		#_kl_loss = self.kl_loss(mu, logvar)
		#z_im = torch.cat([z_im,z], dim=1)
		return z_im#, _kl_loss #, mu, logvar

