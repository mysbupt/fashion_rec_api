import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
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

import requests

import MySQLdb
import simplejson as json
from io import StringIO

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import SimpleStatement
from cassandra.query import ValueSequence

import random
random.seed(1234)
import yaml

use_gpu = GPU_MODE
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)

total_cnt_cache = {}

class PredictData(Dataset):
    def __init__(self, config_path="./config.yaml"):
        conf = yaml.load(open(config_path))
        self.conn_mysql = self.create_connection_mysql(conf["mysql"])
        #self.conn_cassandra = self.create_connection_cassandra(conf["cassandra"])
        mysql_c = self.conn_mysql.cursor()
    
        no_label_ids_query = "SELECT id FROM images WHERE (is_face_in_body is true) AND (num_of_person BETWEEN 1 AND 1) AND (num_of_face BETWEEN 1 AND 1) AND (body_h_percent BETWEEN 0.5 AND 1) AND (face_body_h_percent BETWEEN 0 AND 0.25) AND (label_y_n_ns IS NULL)"
    
        mysql_c.execute(no_label_ids_query)
        no_label_ids = mysql_c.fetchall()
        self.img_ids = []
        for i in no_label_ids:
            img_id = i[0]
            self.img_ids.append(img_id)

        self.image_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def create_connection_cassandra(self, conf):
        hostname = conf["host"]
        keyspace = conf["keyspace"]
        nodes = [hostname]
        cluster = Cluster(nodes)
        session = cluster.connect(keyspace)
        return session
    
    def create_connection_mysql(self, conf):
        db_name = conf["db_name"]
        HOSTNAME = conf["host"]
        USERNAME = conf["username"]
        PASSWD = conf["password"]
        try:
            conn = MySQLdb.connect(host=HOSTNAME, user=USERNAME, passwd=PASSWD , db=db_name)
            return conn
        except:
            print("connection to mysql error")
            exit()
    
    #def get_image(self, conn_cassandra, image_id):
    #    CQL_str = "SELECT image FROM images WHERE img_url_md5 = %s"
    #    rows = conn_cassandra.execute(CQL_str, [img_id])
    #    return rows[0][0]
    def get_image(self, image_id):
        return requests.get("http://172.28.176.132:2222/images/"+image_id+".jpg").content

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.get_image(img_id) 
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = self.image_transform(Image.fromarray(img))
        return img, img_id


def predict(model, dset_loaders):
    preds = []
    ori_imgs = []
    total_batches = len(dset_loaders)
    for i, data in enumerate(dset_loaders):
        print("predict batch %d: %d" %(total_batches, i))
        imgs, img_paths = data
        imgs = Variable(imgs.float().cuda())

        outputs = model(imgs)
        outputs = nn.sigmoid(outputs)
        preds += outputs.data.cpu().numpy().tolist()
        ori_imgs += img_paths

    return preds, ori_imgs


def main():
    predict_set = PredictData()
    dset_loaders = DataLoader(predict_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=20)
    dset_sizes = len(dset_loaders) 
    model_ft = torch.load("fine_tuned_best_model.pt")
    model_ft.cuda()
    
    preds, ori_imgs = predict(model_ft, dset_loaders)
    output = open("./predicts_sigmoid_no_label_imgs.txt", "w")
    for pred, ori_img in sorted(zip(preds, ori_imgs), key=lambda i: i[0], reverse=True):
        output.write(ori_img + "," + ",".join([str(i) for i in pred]) + "\n")


if __name__ == "__main__":
    main()
