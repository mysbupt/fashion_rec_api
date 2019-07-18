import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval
from datetime import datetime

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	parser.add_argument('--dataset', default="csv", help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
	parser = parser.parse_args(args)

	# Create the data loaders
	if parser.csv_train is None:
		raise ValueError('Must provide --csv_train')

	if parser.csv_classes is None:
		raise ValueError('Must provide --csv_classes')

	dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

	if parser.csv_val is None:
		raise ValueError('Must provide --csv_val')
	else:
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=8, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=20, collate_fn=collater, batch_sampler=sampler)

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=10, collate_fn=collater, batch_sampler=sampler_val)
	print("data initialize finish")

	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	retinanet.training = True
	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
	loss_hist = collections.deque(maxlen=500)
	retinanet.train()
	retinanet.module.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.epochs):
		retinanet.train()
		retinanet.module.freeze_bn()
		epoch_loss = []
		
		for iter_num, data in enumerate(dataloader_train):
			optimizer.zero_grad()
			classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

			classification_loss = classification_loss.mean()
			regression_loss = regression_loss.mean()
			loss = classification_loss + regression_loss
			
			if bool(loss == 0):
				continue

			loss.backward()
			torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
			optimizer.step()
			loss_hist.append(float(loss))
			epoch_loss.append(float(loss))

			if iter_num % 500 == 0:
				print(datetime.now(), 'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
				#if parser.dataset == 'csv' and parser.csv_val is not None and iter_num != 0:
				#	print('Evaluating dataset')
				#	mAP, overall_mAP = csv_eval.evaluate(dataset_val, retinanet)
				#	retinanet.train()
				#	retinanet.module.freeze_bn()
				#	print("Epoch %d: overall mAP is: %f\n\n" %(epoch_num, overall_mAP))

			del classification_loss
			del regression_loss

		
		print(datetime.now(), 'Evaluating dataset')
		mAP, overall_mAP = csv_eval.evaluate(dataset_val, retinanet)
		print(datetime.now(), "Epoch %d: overall mAP is: %f\n\n" %(epoch_num, overall_mAP))

		scheduler.step(np.mean(epoch_loss))	
		torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

	retinanet.eval()
	torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
