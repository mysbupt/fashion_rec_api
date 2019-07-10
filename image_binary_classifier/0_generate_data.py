#!/usr/bin/env python

import glob
import MySQLdb
import simplejson as json
from StringIO import StringIO

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import SimpleStatement
from cassandra.query import ValueSequence

import random
random.seed(1234)

import yaml

total_cnt_cache = {}

def create_connection_cassandra(conf):
    hostname = conf["host"]
    keyspace = conf["keyspace"]
    nodes = [hostname]
    cluster = Cluster(nodes)
    session = cluster.connect(keyspace)
    return session


def create_connection_mysql(conf):
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


def get_images(conn_cassandra, image_ids, save_path):
    for img_id in image_ids:
        CQL_str = "SELECT image FROM images WHERE img_url_md5 = %s"
        rows = conn_cassandra.execute(CQL_str, [img_id])
        output = open(save_path + "/" + str(img_id) + ".jpg", "wb")
        output.write(rows[0][0])
        output.close()


def get_current_img_list(data_paths):
    img_set = set()
    for data_path in data_paths:
        for img_file in glob.glob(data_path):
            img_id = img_file.split("/")[-1].split(".")[0]
            img_set.add(img_id)
    return img_set


def main():
    conf = yaml.load(open("./config.yaml"))
    conn_mysql = create_connection_mysql(conf["mysql"])
    conn_cassandra = create_connection_cassandra(conf["cassandra"])
    mysql_c = conn_mysql.cursor()

    # get positive results
    pos_res_ids = "SELECT id FROM images WHERE label_y_n_ns = 1;"
    current_pos_res_ids = get_current_img_list(["./data/train/pos/*.jpg", "./data/val/pos/*.jpg"])
    
    #try:
    mysql_c.execute(pos_res_ids)
    #except:
    #    print pos_res_ids
    #    exit()
    pos_res = mysql_c.fetchall()
    pos_ids = []
    for i in pos_res:
        img_id = i[0]
        if img_id not in current_pos_res_ids:
            pos_ids.append(img_id)
        

    # get negtive results
    neg_res_ids = "SELECT id FROM images WHERE label_y_n_ns = 0;"
    current_neg_res_ids = get_current_img_list(["./data/train/neg/*.jpg", "./data/val/neg/*.jpg"])
    #try:
    mysql_c.execute(neg_res_ids)
    #except:
    #    print neg_res_ids
    #    exit()
    neg_res = mysql_c.fetchall()
    neg_ids = []
    for i in neg_res:
        img_id = i[0]
        if img_id not in current_neg_res_ids:
            neg_ids.append(img_id)
        

    random.shuffle(pos_ids)
    train_pos_len = int(0.9 * (len(pos_ids)))
    print("train_pos_len: %d, val_pos_len: %d" %(train_pos_len, len(pos_ids) - train_pos_len))
    train_pos = pos_ids[0:train_pos_len]
    val_pos = pos_ids[train_pos_len:]

    random.shuffle(neg_ids)
    train_neg_len = int(0.9 * (len(neg_ids)))
    print("train_neg_len: %d, val_neg_len: %d" %(train_neg_len, len(neg_ids) - train_neg_len))
    train_neg = neg_ids[0:train_neg_len]
    val_neg = neg_ids[train_neg_len:]

    print("start to write validation positive: ")
    get_images(conn_cassandra, val_pos, "./data/val/pos")
    print("start to write validation negtive: ")
    get_images(conn_cassandra, val_neg, "./data/val/neg")
    print("start to write train positive: ")
    get_images(conn_cassandra, train_pos, "./data/train/pos")
    print("start to write train negtive: ")
    get_images(conn_cassandra, train_neg, "./data/train/neg")


if __name__ == "__main__":
    main()
