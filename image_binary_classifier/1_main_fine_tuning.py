
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import copy
import os
import glob
from PIL import Image
from fine_tuning_config_file import *
import datetime

NOW = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = "./logs/finetune_%s" %(NOW)
sys.stdout = open(log_file, "w")


if USE_TENSORBOARD:
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=TENSORBOARD_SERVER)
    try:
        cc.remove_experiment(EXP_NAME)
    except:
        pass
    foo = cc.create_experiment(EXP_NAME)


use_gpu = GPU_MODE
if use_gpu:
    print("use GPU")
    torch.cuda.set_device(CUDA_DEVICE)

count = 0

class BinaryData(Dataset):
    def __init__(self, data_path="", mode="train"):
        self.imgs = []
        for img in glob.glob(os.path.join(data_path, "pos/*")):
            self.imgs.append({"path": img, "label": 1})
        for img in glob.glob(os.path.join(data_path, "neg/*")):
            self.imgs.append({"path": img, "label": 0})

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.image_transform = data_transforms[mode]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]["path"]
        label = self.imgs[idx]["label"] 
        img = self.image_transform(Image.open(img_path).convert("RGB"))
        label = torch.tensor([label])
        return img, label



def train_model(model, dset_loaders, dset_sizes, optimizer, lr_scheduler, num_epochs=10):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                mode='train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()
                mode='val'

            running_loss = 0.0
            running_corrects = 0

            counter=0
            # Iterate over data.
            for data in dset_loaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    criterion = nn.BCEWithLogitsLoss()

                optimizer.zero_grad()
                outputs = model(inputs)
                preds = (outputs.data > 0).float()
                
                loss = criterion(outputs, labels)
                counter+=1

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            print('trying epoch loss')
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / float(dset_sizes[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss', epoch_loss, step=epoch)
                    foo.add_scalar_value('epoch_acc', epoch_acc, step=epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ', best_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('returning and looping back')
    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def main():
    data_dir = DATA_DIR
    dsets = {x: BinaryData(os.path.join(data_dir, x), x) for x in ['train', 'val']}
    dset_loaders = {x: DataLoader(dsets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=10) for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    
    if use_gpu:
        #criterion.cuda()
        model_ft.cuda()
    
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    model_ft = train_model(model_ft, dset_loaders, dset_sizes, optimizer_ft, exp_lr_scheduler, num_epochs=20)
    
    params_save_path = "./params/fine_tuned_best_model_weight_%s.pt" %(NOW)  
    torch.save(model_ft, params_save_path)


if __name__ == "__main__":
    main()
