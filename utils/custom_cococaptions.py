from PIL import Image
import os
import os.path
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms

class CocoCaptions(VisionDataset):
	# Returns images at two different resolutions
	def __init__(self, root, annFile, transform_gan=None, transform_vgg=None,  target_transform=None, transforms=None):
		super(CocoCaptions, self).__init__(root, None, None, None)
		from pycocotools.coco import COCO
		self.coco = COCO(annFile)
		self.ids = list(sorted(self.coco.imgs.keys()))
		self.transform_gan = transform_gan
		self.transform_vgg = transform_vgg

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: Tuple (image, target). target is a list of captions for the image.
		"""
		coco = self.coco
		img_id = self.ids[index]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)
		target = [ann['caption'] for ann in anns]

		path = coco.loadImgs(img_id)[0]['file_name']

		img = Image.open(os.path.join(self.root, path)).convert('RGB')

		if self.transform_gan is not None:
			img_gan = self.transform_gan(img)

		if self.transform_vgg is not None:
			img_vgg = self.transform_vgg(img)	

		return img_gan, img_vgg, target


	def __len__(self):
		return len(self.ids)

