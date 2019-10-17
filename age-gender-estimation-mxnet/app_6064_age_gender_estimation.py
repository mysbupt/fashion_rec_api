#!/usr/bin/env python2.7

import os
from flask import Flask, request, json

import cv2 
import sys
import argparse
import numpy as np
import face_model
from datetime import datetime

from skin_detector import get_skin_color, get_color_name, get_rgb_name_file

app = Flask(__name__)


parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/m1/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn or essh option, 0 means mtcnn, 1 means essh')
args = parser.parse_args()
model = face_model.FaceModel(args)

color_name_file = 'color_name.txt'
rgb_file = 'RGB.txt'
rgb_list, name_dict = get_rgb_name_file(rgb_file, color_name_file)

@app.route('/age_gender_estimation', methods=['POST'])
def age_gender_estimation():
    res = {
        'msg': 'success',
        'result': {},
        'T_F': False
    }

    if 'image' not in request.files:
        res['msg'] = 'error: no image uploaded'
        return app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )
    else:
        image = request.files['image']

    img_url_md5 = image.filename.split('/')[-1].split('.')[0]
    img_cont = image.stream.read()
    img = np.fromstring(img_cont, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    img_db, bbox, points = model.get_input(img, args)
    for _ in range(1):
        gender, age = model.get_ga(img_db)

    #skin_color_res = []
    #print("start cal skin color: ", datetime.now())
    #for b in bbox:
    #    face_img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    #    skin_color_rgb = get_skin_color(face_img)
    #    skin_color_name = get_color_name(rgb_list, name_dict, skin_color_rgb)
    #    skin_color_res.append(skin_color_name)
    #print("end of calculating skin color: ", datetime.now())

    if len(bbox) > 0:
        res["T_F"] = True
        res['result'] = {
            'bbox': bbox.tolist(), 
            'points': points.tolist(), 
            'gender': gender.tolist(), 
            'age': age.tolist(),
            #'skin_color': skin_color_res
        }

    return app.response_class(
        response = json.dumps(res),
        status = 200,
        mimetype = 'application/json'
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6064, threaded=False)
