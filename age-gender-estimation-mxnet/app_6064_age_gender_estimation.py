#!/usr/bin/env python2.7

import os
from flask import Flask, request, json

import cv2 
import sys
import argparse
import numpy as np
import face_model


app = Flask(__name__)


parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/m1/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn or essh option, 0 means mtcnn, 1 means essh')
args = parser.parse_args()
model = face_model.FaceModel(args)


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

    if len(bbox) > 0:
        res["T_F"] = True
        res['result'] = {
            'bbox': bbox.tolist(), 
            'points': points.tolist(), 
            'gender': gender.tolist(), 
            'age': age.tolist()
        }

    return app.response_class(
        response = json.dumps(res),
        status = 200,
        mimetype = 'application/json'
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6064, threaded=False)
