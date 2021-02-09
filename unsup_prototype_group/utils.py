import scipy.optimize
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torchvision import transforms


def set_seed(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def save_args(args, writer):
	"""
	Create a txt file in the tensorboard writer directory containing all args.
	:param args:
	:param writer:
	:return:
	"""
	# store args as txt file
	with open(os.path.join(writer.log_dir, 'args.txt'), 'w') as f:
		for arg in vars(args):
			f.write(f"\n{arg}: {getattr(args, arg)}")


def write_imgs(writer, epoch, imgs, tag):
	"""
	Add the reconstructed and original image to tensorboard writer.
	:param writer:
	:param epoch:
	:param imgs:
	:param tag:
	:return:
	"""
	for j in range(10):

		fig = plt.figure()
		ax = plt.axes()
		img = imgs[j].squeeze().detach().cpu()
		ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
		writer.add_figure(f"Val_Sample_{j}/{tag}", fig, epoch, close=True)

		plt.close()


def write_switch_prototypes(writer, epoch, imgs, recon_protos, recon_imgs, switched_rec_proto):
	"""
	Plot the switched reconstructed prototypes, where the attribute of two prototypes has been switched.
	:param writer:
	:param epoch:
	:param imgs:
	:param recon_protos:
	:param recon_imgs:
	:param switched_rec_proto:
	:return:
	"""

	for j in range(10):
		fig = plt.figure()
		ax = plt.axes()
		img = imgs[j].squeeze().detach().cpu()
		ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
		writer.add_figure(f"Val_Switched/Img{j}", fig, epoch, close=True)

		fig = plt.figure()
		ax = plt.axes()
		img = recon_imgs[j].squeeze().detach().cpu()
		ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
		writer.add_figure(f"Val_Switched/Recon_Img{j}", fig, epoch, close=True)

		fig = plt.figure()
		ax = plt.axes()
		img = recon_protos[j].squeeze().detach().cpu()
		ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
		writer.add_figure(f"Val_Switched/Recon_Proto{j}", fig, epoch, close=True)

		fig = plt.figure()
		ax = plt.axes()
		img = switched_rec_proto[j].squeeze().detach().cpu()
		ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
		writer.add_figure(f"Val_Switched/Switched_Recon_Proto{j}", fig, epoch, close=True)

		plt.close()
