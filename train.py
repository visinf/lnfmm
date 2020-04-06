from __future__ import print_function
import os
import random
import numpy as np
import nltk
import argparse
import itertools
import pickle
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from build_vocab_coco import Vocabulary

from text_modules import TextEncoder, TextDecoder
from latent_align_modules import FlowLatent, GaussianDiag
from custom_cococaptions import CocoCaptions


torch.manual_seed(149)
np.random.seed(149)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
	
def load_dataset(data_path,config_setting):
	train_path = str(data_path)+'/train2014'
	val_path =  str(data_path)+'/val2014'
	train_annotations = str(data_path)+'/annotations/captions_train2014.json'
	val_annotations = str(data_path)+'/annotations/captions_val2014.json'

	im_size = 256 if config_setting == 'params_t2i' else 64
	print('data_path:',data_path)
	dataset = CocoCaptions(root=train_path, annFile = train_annotations,
				 transform_gan=transforms.Compose([
				transforms.Resize(im_size), transforms.CenterCrop(im_size),
				transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
				 ]),
				transform_vgg = transforms.Compose([ transforms.RandomResizedCrop(224), transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
				])
	)
	assert dataset

	if config_setting != 'params_t2i':
		imkeyfile = str(data_path)+'/coco_test.txt'
		imkeyfile_ = open(imkeyfile,'r').read().split('\n')[:5000]
		imkeylist = []
		for images in imkeyfile_:
			imagekey = os.path.splitext(os.path.basename(images))[0]
			imagekey  = int(imagekey.split('_')[-1])
			imkeylist.append(imagekey)


		datasetval = CocoCaptions(root=val_path, annFile = val_annotations,
					 transform_gan=transforms.Compose([
					transforms.Resize(im_size), transforms.CenterCrop(im_size), 
					transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
					 ]),
					transform_vgg = transforms.Compose([ transforms.RandomResizedCrop(224), transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					])
		)

		assert datasetval

		fullval = datasetval.ids
		datasetval.ids = [x for x in fullval if x not in imkeylist]

		dataset = torch.utils.data.ConcatDataset([dataset,datasetval])

	return dataset



def get_properseq(x, img_gan, img_vgg):
	sorted_imgs_gan = torch.zeros(img_gan.size())
	sorted_imgs_vgg = torch.zeros(img_vgg.size())
	seq_lengths = np.asarray([len(c) for c in x])
	sort_order = np.argsort(seq_lengths)[::-1]
	seq = np.zeros((len(x),max_length))
	for idx in range(len(seq)):
		a = sort_order[idx]
		seq[idx,0:seq_lengths[a]] = x[a]
		sorted_imgs_gan[idx] = img_gan[a]
		sorted_imgs_vgg[idx] = img_vgg[a]

	seq = seq.astype(np.float64)
	seq_lengths = seq_lengths[sort_order].astype(np.int32)
	seq = torch.LongTensor(seq.astype(np.float64)).to(device)
	seq_lengths = torch.LongTensor(seq_lengths.astype(np.int32)).to(device) 
	return seq, seq_lengths, sort_order, sorted_imgs_gan, sorted_imgs_vgg



def get_sentence_targets(cap):
	target = []
	for k in range(batch_size):
		#for j in xrange(5):
		j = np.random.randint(0,5)
		tokens = nltk.tokenize.word_tokenize(str(cap[j][k]).lower())
		caption = []
		caption.append(vocab_w2i['<start>'])
		caption.extend([vocab_w2i[token] for token in tokens if token in vocab_w2i.keys()])
		caption = caption[0:(max_length-1)]
		caption.append(vocab_w2i['<end>'])

		target.append(caption)
	return target


def get_sentence_NLL_loss(seq,logp):  
	target = seq[:,1:]
	target = target[:, :torch.max(seq_lengths).item()].contiguous().view(-1)
	preds = logp.clone().detach().cpu().numpy()
	logp = logp.view(-1, logp.size(2))
	return NLL(logp, target), preds

def adjust_learning_rate(optimizer, epoch,lr):
	"""Sets the learning rate to the initial LR decayed by 0.5 every 30 epochs"""
	lr = lr * (0.5 ** (epoch // 25))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr





def autoencode_image( encoder, decoder, flow_latent_image, im_batch_gan, im_batch_vgg, z_im_text2img ):
	'''Build image pipeline '''
	if config_setting == 'params_i2t': 
		encoded = encoder(im_batch_vgg)
	else:
		encoded = encoder(im_batch_gan)
	_encoded = encoded[:,:] + torch.randn(encoded.size(),device=device)*0.05
	encoded[:,img_dim:] = encoded[:,img_dim:] + torch.randn(batch_size,noise_im,device=device)*0.05 
	decoded = decoder(t2i = _encoded[:,:img_dim], z = encoded[:,img_dim:])

	rec_loss = torch.sum(torch.abs( im_batch_gan - decoded ), dim=(1,2,3)) / batch_size

	_, nll, _ = flow_latent_image(encoded[:,img_dim:].to(device), cond=z_im_text2img, reverse=False)

	return encoded, rec_loss, decoded ,nll


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='params_i2t', help='Experiment settings.')
	args = parser.parse_args()
	config_setting = args.config

	''' Get parameters from params.json'''
	config = json.loads(open('params.json', 'r').read())
	config = config[config_setting]
	data_path = config['pathToData']
	vocab_path = config['vocab_path']
	noise_im = int(config['noise_im'])
	noise_txt = int(config['noise_txt'])
	embed_size = int(config['embed_size'])
	max_length = int(config['max_length'])
	num_encoder_tokens = int(config['num_encoder_tokens'])
	num_decoder_tokens= int(config['num_decoder_tokens'])
	word_dim = int(config['word_dim'])
	batch_size = int(config['batch_size'])
	num_gpus = int(config['num_gpus'])
	img_dim = int(config['img_dim'])
	epochs = int(config['epochs'])
	lambda_1 = config['lambda_1']
	lambda_2 = config['lambda_2']
	lambda_3 = config['lambda_3']
	lambda_4 = config['lambda_4']
	lambda_5 = config['lambda_5']
	lambda_5_G = config['lambda_5_G']
	chkpt_interval = config['chkpt_interval']
	lr_i_e = config['lr_i_e']
	beta_1_i_e = config['beta_1_i_e']
	beta_2_d = config['beta_2_d']
	device = torch.device("cuda:0")

	'''load vocabulary from vocab.pkl'''

	vocab_class = pickle.load( open(vocab_path,'rb'))
	vocab = vocab_class.idx2word
	vocab_w2i = vocab_class.word2idx
	vocab_word2vec = []#[None for _ in xrange(len(vocab))]
	vocab_wordlist = [];
	for w_id,word in tqdm(vocab.items()):
		vocab_wordlist.append(word)

	vocab_size = len(vocab_wordlist)

	'''set data loaders'''
	
	dataset = load_dataset(data_path,config_setting)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(4), drop_last=True)
	dataloader_iterator = iter(dataloader)


	'''define models'''
	if config_setting == 'params_i2t':
		from SaGAN import Generator64 as Generator
		from SaGAN import Discriminator64 as Discriminator
		from image_modules import VGG16Encoder as ImageEncoder
		image_encoder = ImageEncoder( img_dim + noise_im, hidden_dims=2048 ).cuda()

	else:
		from SaGAN import Generator 
		from SaGAN import Discriminator
		from image_modules import ImageEncoder
		image_encoder = ImageEncoder( img_dim + noise_im, noise_im).cuda()


	image_decoder = Generator( z_dim=noise_im, g_conv_dim=32, t2i_dim = img_dim)
	image_encoder =  nn.DataParallel(image_encoder).cuda()
	image_decoder =  nn.DataParallel(image_decoder).cuda()

	txtEncoder = TextEncoder(batch_size//num_gpus,word_dim,embed_size,vocab_size)
	txtEncoder = nn.DataParallel(txtEncoder).cuda()
	txtDecoder = TextDecoder(batch_size,embed_size,vocab_size)
	txtDecoder = (txtDecoder).cuda()

	flow_text_cond = FlowLatent(batch_size=batch_size,input_dim=noise_txt,hidden_channels=1024,K=16,gaussian_dims=noise_txt,gaussian_var=0.25,cond_dim=img_dim,coupling='full')
	flow_text_cond = flow_text_cond.to(device)

	flow_latent_align = FlowLatent(batch_size//num_gpus,img_dim,hidden_channels=1024,K=12,gaussian_dims=(0),gaussian_var=0,coupling='linear').cuda()
	flow_latent_align= nn.DataParallel(flow_latent_align).cuda()
	flow_latent_image = FlowLatent(batch_size,noise_im,hidden_channels=512,gaussian_dims=noise_im,gaussian_var=0.25,cond_dim=img_dim,coupling='full').cuda()
	disc = Discriminator(d_conv_dim=32, t2i_dim = img_dim)
	disc =  nn.DataParallel(disc).cuda()


	'''define optimizers'''
	optimizerG_align = optim.Adam(flow_latent_align.parameters(),lr=0.0001)
	optimizerG_image = optim.Adam(flow_latent_image.parameters(),lr=0.0001)
	optimizerI_e = optim.Adam(image_encoder.parameters(),lr=lr_i_e,betas=(beta_1_i_e, 0.999))#, betas=(0.0, 0.9)
	optimizerI_d = optim.Adam(image_decoder.parameters(),lr=0.0001, betas=(0.0, 0.999))#, betas=(0.0, 0.9)
	optimizerF = optim.Adam(itertools.chain(txtEncoder.parameters(),txtDecoder.parameters()),lr=0.0001)
	optimizerG_cond_text = optim.Adam(flow_text_cond.parameters(),lr=0.0001)
	optimizerD = optim.Adam(disc.parameters(),lr=0.0003, betas=(0.0, beta_2_d))
	''''''

	NLL = nn.NLLLoss(size_average=False, ignore_index=0)
	GAN_loss = nn.BCELoss()



	count = 0;
	err_D = None
	true_labels = torch.ones(batch_size,).float().cuda()
	fake_labels = torch.zeros(batch_size,).float().cuda()

	discriminator_iter = 2


	for epoch in range(epochs):
		adjust_learning_rate(optimizerI_e, epoch, 0.00001)
		train_bar = tqdm(range(len(dataset)//batch_size))
		for i in train_bar: 

			txtEncoder.zero_grad()
			txtDecoder.zero_grad()
			flow_latent_image.zero_grad()
			flow_latent_align.zero_grad()
			image_encoder.zero_grad()
			image_decoder.zero_grad()
			disc.zero_grad()
			flow_text_cond.zero_grad()
			optimizerF.zero_grad()
			optimizerG_image.zero_grad()
			optimizerG_align.zero_grad()
			optimizerI_e.zero_grad()
			optimizerI_d.zero_grad()
			optimizerD.zero_grad()
			optimizerG_cond_text.zero_grad()

			# Get Data
			try:
				data = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(dataloader)  
				data = next(dataloader_iterator)

			img_gan, img_vgg, captions = data # cap has shape batch_size*10
			captions_batch = get_sentence_targets(captions)
			seq, seq_lengths, sort_array, img_gan, img_vgg = get_properseq(captions_batch, img_gan, img_vgg)
			txtencoded_hidden = txtEncoder(seq,seq_lengths)
			txtencoded_hidden =  txtencoded_hidden+ 0.05*torch.randn(txtencoded_hidden.size()).to(device)
			txtencoded_hidden[:,img_dim:] =  txtencoded_hidden[:,img_dim:] + 0.10*torch.randn(txtencoded_hidden[:,img_dim:].size()).to(device)
			seq_lengths = seq_lengths - 1;

			if i%discriminator_iter == 0:
				logp = txtDecoder(seq,txtencoded_hidden,seq_lengths)
				NLL_loss, preds = get_sentence_NLL_loss(seq,logp)
			
			z, nll, _ = flow_latent_align(x=txtencoded_hidden[:,:img_dim].to(device), z_im=None, z=None, cond=None, eps_std=None, reverse=False) 
			z_im_text2img = z[:,:img_dim]

			z_im_full, image_rec_loss, z_im_true,nll_im = autoencode_image( image_encoder, image_decoder, flow_latent_image, img_gan.cuda(), img_vgg.cuda(), z_im_text2img )
			z_im_full = z_im_full[:,:].cuda()
			z_im = z_im_full[:,:img_dim]
			z_rev, _ = flow_latent_align(x=z_im.to(device), z_im=None, z=None, cond=None, eps_std=None, reverse=True)
			z_text, nll_text_cond, _  = flow_text_cond(x=txtencoded_hidden[:,img_dim:].to(device), z_im=None, z=None, cond=z_rev[:,:img_dim].to(device), eps_std=None, reverse=False)
			
			if i%discriminator_iter == 0:
				_z_rev = torch.cat((z_rev,txtencoded_hidden[:,img_dim:]),dim=1)
				logp = txtDecoder(seq,_z_rev.cuda(),seq_lengths)
				img2txt_loss, preds_inf_latent = get_sentence_NLL_loss(seq,logp)


			cond_loss_forward = F.mse_loss(z[:,:img_dim],z_im.to(device))
			if i%discriminator_iter == 0:
				z_im_decoded = image_decoder(t2i = z_im_text2img, z = z_im_full[:,img_dim:] )#z_im_text2img.view(z_im_text2img.size(0),z_im_text2img.size(1),1,1))#.clone().detach()
				text2img_loss = torch.sum(torch.abs( img_gan.cuda() - z_im_decoded.cuda() ), dim=(1,2,3)) / batch_size
				

			if i%discriminator_iter != 0:

				rev, _ = flow_latent_image(x=None,z_im=None, z=None, cond=z_im_text2img.to(device), eps_std=None, reverse=True)
				rev[torch.isnan(rev)] = 0
				z_im_decoded = image_decoder(t2i = z_im_text2img, z = rev )
				out_fake = disc(z=z_im_decoded.detach(),t2i =txtencoded_hidden[:,:img_dim].detach()).view(-1).cuda()
				err_D_fake_tx = nn.ReLU()(1.0 - out_fake).mean()

				try:
					data = next(dataloader_iterator)
				except StopIteration:
					dataloader_iterator = iter(dataloader)  
					data = next(dataloader_iterator)

				img_gan, _, _ = data
				out_real = disc(z=img_gan.float().to(device),t2i=txtencoded_hidden[:,:img_dim].detach() ).view(-1).cuda()
				err_D_real = nn.ReLU()(1.0 + out_real).mean()
				err_D = (err_D_fake_tx + err_D_real).to(device)
				err_D.backward()
				optimizerD.step()

			else: 
				err_G_tx = disc(z=z_im_decoded, t2i=txtencoded_hidden[:,:img_dim].detach()).view(-1).cuda()
				err_G = err_G_tx
				loss_shared_dim = lambda_1*(1.5*torch.mean(text2img_loss)+ 0.6*torch.mean(img2txt_loss)+ 1000.0*torch.mean(nll))
				loss_text_lflow = lambda_2*torch.mean(nll_text_cond)
				loss_im_lflow = lambda_3*torch.mean(nll_im)
				loss_txt_rec = lambda_4*torch.mean(NLL_loss)
				loss_im_rec  = lambda_5*(torch.mean(image_rec_loss)+lambda_5_G*torch.mean(err_G))

				loss = (loss_shared_dim+loss_text_lflow+loss_im_lflow+loss_txt_rec+loss_im_rec).to(device)
				loss.backward()


				torch.nn.utils.clip_grad_value_(txtEncoder.parameters(), 1.0)
				torch.nn.utils.clip_grad_value_(flow_latent_align.parameters(), 1.0)
				torch.nn.utils.clip_grad_value_(flow_latent_image.parameters(), 5.0)
				torch.nn.utils.clip_grad_value_(txtDecoder.parameters(), 1.0)
				torch.nn.utils.clip_grad_value_(flow_text_cond.parameters(), 1.0)


				optimizerF.step()
				optimizerG_align.step()
				optimizerG_image.step()
				optimizerI_e.step()
				optimizerI_d.step()
				optimizerG_cond_text.step()

				train_bar.set_description('Loss %.2f | Epoch %d -- Iteration ' % (loss.item(),epoch))

			if i%chkpt_interval==0:

				torch.save({
							'image_encoder_sd': image_encoder.module.state_dict(),
							'image_decoder_sd': image_decoder.module.state_dict(),
							'flow_latent_align_sd': flow_latent_align.module.state_dict(),
							'flow_latent_image_sd': flow_latent_image.state_dict(),
							'flow_text_cond_sd': flow_text_cond.state_dict(),
							'txtEncoder_sd': txtEncoder.module.state_dict(),
							'txtDecoder_sd': txtDecoder.state_dict(),
							'disc_sd': disc.module.state_dict(),
							'optimizer_image_encoder_sd': optimizerI_e.state_dict(),
							'optimizer_image_decoder_sd': optimizerI_d.state_dict(),
							'optimizer_flow_latent_align_sd': optimizerG_align.state_dict(),
							'optimizer_flow_latent_image_sd': optimizerG_image.state_dict(),
							'optimizer_flow_text_cond_sd': optimizerG_cond_text.state_dict(),
							'optimizer_txt_sd': optimizerF.state_dict(),
							'optimizerD_sd' : optimizerD.state_dict()
							}, './model_checkpoint_t2i.pt')





