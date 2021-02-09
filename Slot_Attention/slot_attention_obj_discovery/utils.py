import scipy.optimize
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms


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


def write_recon_imgs_plots(writer, epoch, recon, data):
	"""
	Add the reconstructed and original image to tensorboard writer.
	:param writer:
	:param epoch:
	:param recon:
	:param data:
	:param i:
	:return:
	"""
	for j in range(1):
		fig = plt.figure()
		ax = plt.axes()
		img = recon[j].squeeze().detach().cpu()
		# unnormalize images
		img = img / 2. + 0.5  # Rescale to [0, 1].
		ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
		# ax.imshow(img.numpy())
		writer.add_figure(f"Sample_{j}/Recon", fig, epoch, close=True)
		fig = plt.figure()
		ax = plt.axes()
		img = data[j].squeeze().detach().cpu()
		# unnormalize images
		img = img / 2. + 0.5  # Rescale to [0, 1].
		ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
		# ax.imshow(img.numpy())
		writer.add_figure(f"Sample_{j}/Orig", fig, epoch, close=True)


def write_slot_imgs(writer, epoch, recons):
	"""
	Add the individual slots to tensorboard writer.
	:param writer:
	:param epoch:
	:param recons:
	:param data:
	:param i:
	:return:
	"""
	# `recons` has shape: [batch_size, num_slots, num_channels, width, height].

	for j in range(1):
		slots = recons[j].squeeze().detach().cpu()

		fig, axs = plt.subplots(3, 4)
		# fig, axs = plt.subplots(2, 2)
		for i, ax in enumerate(axs.flat):
			if i > 10:
				break
			img = slots[i].squeeze()
			# unnormalize images
			img = img / 2. + 0.5  # Rescale to [0, 1].
			ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
			# ax.imshow(img.numpy())
			ax.set_title(f"Slot {i}")
		writer.add_figure(f"Sample_{j}/Slot_Recons", fig, epoch, close=True)


def write_mask_imgs(writer, epoch, masks):
	"""
	Add the individual slots to tensorboard writer.
	:param writer:
	:param epoch:
	:param recons:
	:param data:
	:param i:
	:return:
	"""
	# `recons` has shape: [batch_size, num_slots, 1, width, height].

	for j in range(1):
		mask = masks[j].squeeze().detach().cpu()

		fig, axs = plt.subplots(3, 4)
		for i, ax in enumerate(axs.flat):
			if i > 10:
				break
			img = mask[i].squeeze().numpy()
			ax.imshow(img, cmap='gray')
			ax.set_title(f"Slot {i}")
		writer.add_figure(f"Sample_{j}/Slot_masks", fig, epoch, close=True)


def write_slots(writer, epoch, slots):
	"""
	Add the individual slots to tensorboard writer.
	:param writer:
	:param epoch:
	:param slots:
	:return:
	"""
	# slots has shape: [batch_size, num_slots, slot_size].

	for j in range(1):
		slots = slots[j].squeeze().detach().cpu()

		fig, axs = plt.subplots(3, 4)
		for i, ax in enumerate(axs.flat):
			if i > 10:
				break
			img = slots[i].squeeze()
			img = torch.reshape(img, (8, 8))
			ax.imshow(np.array(transforms.ToPILImage()(img).convert("L")), cmap='gray')
			ax.imshow(img.numpy(), cmap='gray')
			ax.set_title(f"Slot {i}")
		writer.add_figure(f"Sample_{j}/Slots", fig, epoch, close=True)


# def write_attn(writer, epoch, attn):
# 	"""
# 	Add the individual slots to tensorboard writer.
# 	:param writer:
# 	:param epoch:
# 	:param slots:
# 	:return:
# 	"""
# 	# slots has shape: [batch_size, num_slots, slot_size].
#
# 	for j in range(1):
# 		attn = attn[j].squeeze().detach().cpu()
#
# 		fig, axs = plt.subplots(3, 4)
# 		# fig, axs = plt.subplots(2, 2)
# 		for i, ax in enumerate(axs.flat):
# 			if i > 10:
# 				break
# 			img = attn[i].squeeze()
# 			img = img.reshape(128, 128).numpy()
# 			ax.imshow(img.numpy(), cmap='gray')
# 			ax.set_title(f"Slot {i}")
# 		writer.add_figure(f"Sample_{j}/Attn", fig, epoch, close=True)
