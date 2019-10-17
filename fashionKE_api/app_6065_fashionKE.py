#!/usr/bin/env python

import os
from flask import Flask, request, json

import cv2 
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

import sys
sys.path.insert(0, "./models")
from FashionKE import *
from utility import *
import yaml

app = Flask(__name__)


conf = yaml.load(open("./config.yaml"))
conf["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = FashionData(conf)
conf["num_occasion"] = 10
conf["num_cat"] = len(dataset.cat_code)
conf["num_attr"] = len(dataset.attr_code)
conf["num_country"] = len(dataset.country_code) + 1
conf["attr_class_num"] = [0] * conf["num_attr"]
for attr, code in dataset.attr_code.items():
    conf["attr_class_num"][code] = len(dataset.attr_val_code[attr])
model = OccCatAttrClassifier(conf, dataset.word_embedding, dataset.meta_embed, dataset.cat_noise_estimate, dataset.attr_noise_estimate_list)
model.load_state_dict(torch.load("./params/best_model"))
model.to(device=conf["device"])

def get_cat_attr_map():
    data = json.load(open("./data/clothes_category_attribute_value.json"))
    res = {}
    for cat, attr_vals in data.items():
        each_attrs = set()
        for each in attr_vals:
            for attr, vals in each.items():
                each_attrs.add(attr)
        res[cat] = each_attrs
    return res


def get_attr_id_val_map():
    data = json.load(open("./data/code_attr_val.json"))
    res = {}
    for attr, val_id in data.items():
        each_res = {}
        for val, id_ in val_id.items():
            each_res[id_] = val
        res[attr] = each_res
    return res


def get_id_str_map(filepath):
    res = {}
    for occ, id_ in json.load(open(filepath)).items():
        res[id_] = occ
    return res

word_id_map = json.load(open("./data/word_id_map.json"))
id_occ_map = get_id_str_map("./data/code_occasion.json")
id_cat_map = get_id_str_map("./data/code_cat.json")
id_attr_map = get_id_str_map("./data/code_attr.json")
attr_id_val_map = get_attr_id_val_map()
cat_attr_map = get_cat_attr_map() 

@app.route('/fashionKE', methods=['POST'])
def fashionKE():
    result = {
        'msg': 'success',
        'result': {},
        'T_F': True 
    }

    if 'image' not in request.files:
        result['msg'] = 'error: no image uploaded'
    elif 'body_bboxes' not in request.form:
        result['msg'] = 'error: no body_bboxes provided'
    elif 'cloth_bboxes' not in request.form:
        result['msg'] = 'error: no cloth_bboxes provided'
    else:
        image = request.files['image']
        body_bboxes = json.loads(request.form['body_bboxes'])
        cloth_bboxes = json.loads(request.form['cloth_bboxes'])
        if 'text' in request.form:
            text = request.form['text']
        else:
            text = ''
        num_box = len(cloth_bboxes)
    if result['msg'] != 'success':
        return app.response_class(
            response = json.dumps(result),
            status = 200,
            mimetype = 'application/json'
        )

    #print(body_bboxes)
    #print(cloth_bboxes)
    #print(text)

    #body_bboxes = [[
    #    327.0808410644531,
    #    324.7906188964844,
    #    202.45632934570312,
    #    566.84619140625
    #]]
    #cloth_bboxes = [[
    #    204,
    #    138,
    #    429,
    #    421
    #]]
    #text = "I like wedding"

    img_cont = image.stream.read()
    occ_res, cat_res, attr_val_res = predict(model, img_cont, body_bboxes, cloth_bboxes, text)

    occ_id = torch.argmax(occ_res[0]).cpu().numpy().tolist()
    occ = id_occ_map[occ_id]
    occ_score = torch.softmax(occ_res[0], dim=-1)[occ_id].cpu().detach().numpy().tolist()
    #print("occ_id: %d, occ: %s, occ_score: %f" %(occ_id, occ, occ_score))

    cat_ids = torch.argmax(cat_res[0, :num_box, :], dim=-1).cpu().numpy().tolist()
    if num_box == 1:
        cat_ids = [cat_ids]
    cats = [id_cat_map[cat_id] for cat_id in cat_ids]
    cat_scores = [torch.softmax(each_res, dim=-1)[cat_id].cpu().detach().numpy().tolist() for each_res, cat_id in zip(cat_res[0][0:num_box], cat_ids)]
    #print("cat_ids:", cat_ids)
    #print("cats:", cats)
    #print("cat_scores:", cat_scores)

    # note here attr_val_res is a list of attributes, the element size is: [batch_size, num_cloth, num_vals_for_each_attr]
    attr_val_scores = [{} for i in range(num_box)]
    for attr_id, rest in enumerate(attr_val_res):
        # rest: [batch_size, num_cloth, num_vals_for_each_attr]
        attr_str = id_attr_map[attr_id]
        rest = rest[0] # the batch_size is 1 [num_cloth, num_vals_for_each_attr]
        rest = rest[:num_box, :] # drop all the padding results
        val_ids = torch.argmax(rest, dim=-1).cpu().numpy().tolist()
        val_strs = [attr_id_val_map[attr_str][val_id] for val_id in val_ids]
        val_scores = [torch.softmax(rest, dim=-1)[i, val_id].cpu().detach().numpy().tolist() for i, val_id in enumerate(val_ids)]
        for box_id, (cat, val_id, val_str, val_score) in enumerate(zip(cats, val_ids, val_strs, val_scores)):
            if attr_str in cat_attr_map[cat]:
                attr_val_scores[box_id][":".join([attr_str, val_str])] = val_score

    #print(json.dumps(attr_val_scores, indent=4))
    result["result"] = {"clothes": [], "occasion": "%s:%f" %(occ, occ_score)}
    for cloth_box, cat, cat_score, attrval_score in zip(cloth_bboxes, cats, cat_scores, attr_val_scores):
        tags = []
        tags.append({"tag": "category:%s" %(cat), "score": cat_score})
        for attrval, score in sorted(attrval_score.items(), key=lambda i: i[0]):
            tags.append({"tag": attrval, "score": score})
        result["result"]["clothes"].append({"box": cloth_box, "tags": tags})

    return app.response_class(
        response = json.dumps(result),
        status = 200,
        mimetype = 'application/json'
    )


def convert_text2ids(text):
    res = []
    for word in text.strip().split():
        if word not in word_id_map:
            continue
        res.append(word_id_map[word])
    if len(res) < 16:
        res += [0] * (16 - len(res))
    else:
        res = res[:16]
    return res


def predict(model, test_img, body_bboxes, cloth_bboxes, text):
    img_size = (224, 224)
    image_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def crop_imgs(ori_img, bbox):
        img = ori_img.crop(bbox)
        res = image_transform(img)
        return res

    whole_img = np.fromstring(test_img, np.uint8)
    whole_img = cv2.imdecode(whole_img, cv2.IMREAD_COLOR)
    ori_img = Image.fromarray(whole_img).convert("RGB")
    whole_img = image_transform(ori_img).unsqueeze(0)

    max_cloth_num = 5
    imgs = torch.zeros([max_cloth_num, 3, img_size[0], img_size[1]])
    img_loc = []
    for i, (body_box, cloth_box) in enumerate(zip(body_bboxes, cloth_bboxes)):
        # body_center_x, cloth_center_y
        body_center_x = body_box[0] + (body_box[2] - body_box[0]) / 2.0
        cloth_center_y = cloth_box[1] + (cloth_box[3] - cloth_box[1]) / 2.0
        tmp_loc = [i, body_center_x, cloth_center_y]
        img_loc.append(tmp_loc)

    #for i, each in enumerate(self.img_meta_map[img_id]["cloth_body_face"]):
    for x in sorted(img_loc, key=lambda i: (i[1], i[2])):
        # sort the clothes in the order of location, left->right (by different bodies' center), top->down (by different clothes' center)
        i = x[0]
        cloth_bbox = cloth_bboxes[i]
        cloth_img = crop_imgs(ori_img, cloth_bbox)
        imgs[i] = cloth_img
    imgs = imgs.unsqueeze(0)
    text = torch.LongTensor(convert_text2ids(text)).unsqueeze(0)

    ## the following are useless
    season = torch.tensor([0]).to(device=conf["device"])
    age = torch.zeros((1, max_cloth_num), dtype=torch.long).to(device=conf["device"])
    gender = torch.zeros((1, max_cloth_num), dtype=torch.long).to(device=conf["device"])
    country = torch.tensor([0]).to(device=conf["device"])
    ## end of useless

    whole_img = whole_img.to(device=conf["device"])
    imgs = imgs.to(device=conf["device"])
    text = text.to(device=conf["device"])
    occ_res, cat_res, attr_val_res = model.predict(whole_img, imgs, season, age, gender, country, text)

    return occ_res, cat_res, attr_val_res 


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6065)
