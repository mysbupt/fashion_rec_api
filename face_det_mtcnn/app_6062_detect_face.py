#!/usr/bin/env python

import os
import cv2
from flask import Flask, request, json
import tempfile

from demo import *

app = Flask(__name__)


minsize = 20
caffe_model_path = "./model"
threshold = [0.6, 0.7, 0.7]
factor = 0.709

caffe.set_mode_gpu()
PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)


@app.route('/detect_face', methods=['POST'])
def detect_faces():
    res = {
        'msg': 'success',
        'result': [],
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

    tmp_file, image_tmp_path = tempfile.mkstemp()
    image.save(image_tmp_path)
    img_url_md5 = image.filename.split('/')[-1].split('.')[0]

    try:
        img = cv2.imread(image_tmp_path)
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp

        boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
        res['result'] = {'boxes': boundingboxes.tolist(), 'points': points.tolist()}
    except:
        res['msg'] = 'error: inner error'
        os.close(tmp_file)
        os.remove(image_tmp_path)
        return app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )

    res['T_F'] = False
    if len(res['result']) != 0:
        res['T_F'] = True 

    os.close(tmp_file)
    os.remove(image_tmp_path)
    print res

    return app.response_class(
        response = json.dumps(res),
        status = 200,
        mimetype = 'application/json'
    )


def start_webserver():
    app.run(host='0.0.0.0', port=6062)

if __name__ == "__main__":
    start_webserver()
