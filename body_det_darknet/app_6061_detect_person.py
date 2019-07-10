#!/usr/bin/env python

import os
from flask import Flask, request, json
from darknet import *
import tempfile

app = Flask(__name__)


net = load_net("./cfg/yolov3.cfg", "./models/yolov3.weights", 0)
meta = load_meta("./cfg/coco.data")


@app.route('/detect_person', methods=['POST'])
def detect_person():
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
        res['result'] = detect(net, meta, image_tmp_path)
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
    if len(res['result']) == 0:
        res['T_F'] = False
    else:
        for each in res['result']:
            if each[0] == 'person':
                res['T_F'] = True
                break

    os.close(tmp_file)
    os.remove(image_tmp_path)

    return app.response_class(
        response = json.dumps(res),
        status = 200,
        mimetype = 'application/json'
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6061)
