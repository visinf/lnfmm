# Introduction

<p align="center">
  <img width="320" height="200" src="/assets/teaser.png" hspace="30">
</p>

This repository is the PyTorch implementation of the paper:

**Latent Normalizing Flows for Many-to-Many Cross Domain Mappings (ICLR 2020)** 

[Shweta Mahajan](https://www.visinf.tu-darmstadt.de/team_members/smahajan/smahajan.en.jsp), [Iryna Gurevych](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/staff_ukp/prof_dr_iryna_gurevych/index.en.jsp), [Stefan Roth](https://www.visinf.tu-darmstadt.de/team_members/sroth/sroth.en.jsp)

# Getting started

This code has been developed under Python 3.5, Pytorch 1.0.0 and CUDA 10.0.


1. Please run `requirements.py` to check if all required packages are installed.
2. The dataset used in this project is COCO 2014. The dataset is available [here](http://cocodataset.org/#download).

# Training

The script `train.py` is used for training. The parameters are listed in `params.json`. Note that there are two different configurations for best performance on the image captioning and text-to-image synthesis tasks.

Example usage to train a model on COCO 2014 for captioning is,

	python train.py --config params_i2t


Example usage to train a model on COCO 2014 for text-to-image synthesis task is,

	python train.py --config params_t2i

Note that for training CUDA 10.0 and GPU devices are required. The number of GPUs used can be set in `params.json`. Also note that we use 1 Nvidia Volta V100 GPU and 3 Nvidia Volta V100 GPUs with 32GB for the captioning and text-to-image synthetis tasks respectively.

# Generation and Validation

For evalutaion we use the following repos,

1. Oracle - We use the version of pycocoeval cap which supports Python 3 available [here](https://github.com/salaniz/pycocoevalcap). 
2. Concensus Reranking -  We use the repo of [mRNN-CR](https://github.com/mjhucla/mRNN-CR).
3. Diversity - We use the repo of [DiversityMetrics](https://github.com/qingzwang/DiversityMetrics) (requires Python 2.7).

Checkpoints will be available [here](https://drive.google.com/drive/folders/1ZYfajytm7e-aI8KnJRR92BTkddZDaczv?fbclid=IwAR0TFSi6JSl1FlKcF_7Jvz0OFPPZwWLrePRz0t__J_vnu1w_BNIODxUf7bQ) shortly.


# Bibtex

	@inproceedings{mahajan2020latent,
	title = {Latent Normalizing Flows for Many-to-Many Cross-Domain Mappings},
	author = {Mahajan, Shweta and Gurevych, Iryna and Roth, Stefan},
	booktitle = {International Conference on Learning Representations},
	year = {2020},
	}