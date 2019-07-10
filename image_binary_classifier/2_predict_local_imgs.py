from __future__ import print_function, division
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import glob
import copy
import os
from PIL import Image
from fine_tuning_config_file import *

use_gpu = GPU_MODE
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)


class PredictData(Dataset):
    def __init__(self, data_path=""):
        self.img_ids = [] 
        for img in glob.glob(os.path.join(data_path, "*")):
            self.img_ids.append(img)

        self.image_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = self.img_ids[idx]
        #img = open(img_path, "rb")
        #img = np.fromstring(img, np.uint8)
        #img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        #img = self.image_transform(Image.fromarray(img))
        img = self.image_transform(Image.open(img_path).convert("RGB"))
        return img, img_path


def predict(model, dset_loaders):
    preds = []
    ori_imgs = []
    for i, data in enumerate(dset_loaders):
        print("predict batch %d" %(i))
        imgs, img_paths = data
        imgs = Variable(imgs.float().cuda())

        outputs = model(imgs)
        outputs = nn.sigmoid(outputs)
        preds += outputs.data.cpu().numpy().tolist()
        ori_imgs += img_paths

    return preds, ori_imgs


def main():
    # predict positive samples of validation set
    predict_set = PredictData("./data/val/pos")
    dset_loaders = DataLoader(predict_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)
    dset_sizes = len(dset_loaders) 
    model_ft = torch.load("fine_tuned_best_model.pt")
    model_ft.cuda()
    
    preds, ori_imgs = predict(model_ft, dset_loaders)
    output = open("./predicts_pos.txt", "w")
    for pred, ori_img in sorted(zip(preds, ori_imgs), key=lambda i: i[0], reverse=True):
        output.write(ori_img + "," + ",".join([str(i) for i in pred]) + "\n")


    # predict negtive samples of validation set
    predict_set = PredictData("./data/val/neg")
    dset_loaders = DataLoader(predict_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)
    dset_sizes = len(dset_loaders) 
    model_ft = torch.load("fine_tuned_best_model.pt")
    model_ft.cuda()
    
    preds, ori_imgs = predict(model_ft, dset_loaders)
    output = open("./predicts_neg.txt", "w")
    for pred, ori_img in sorted(zip(preds, ori_imgs), key=lambda i: i[0]):
        output.write(ori_img + "," + ",".join([str(i) for i in pred]) + "\n")


if __name__ == "__main__":
    main()
