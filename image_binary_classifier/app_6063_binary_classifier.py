#!/usr/bin/env python

import os
from flask import Flask, request, json
from darknet import *

import cv2 
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


app = Flask(__name__)


#model_ft = torch.load("./params/fine_tuned_best_model.pt")
model_ft = torch.load("./params/fine_tuned_best_model_weight_2018-11-13 10:59:11.pt")
model_ft.cuda()


@app.route('/binary_classifier', methods=['POST'])
def binary_classifier():
    res = {
        'msg': 'success',
        'result': 0.0,
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

    score = predict(model_ft, img_cont)
    res["result"] = score
    res["T_F"] = True

    """
    try:
        res['result'] = predict(model_ft, image_cont)
        res['T_F'] = True
    except:
        res['msg'] = 'error: inner error'
        return app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )
    """

    return app.response_class(
        response = json.dumps(res),
        status = 200,
        mimetype = 'application/json'
    )


def predict(model, test_img):
    image_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = np.fromstring(test_img, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = image_transform(Image.fromarray(img).convert("RGB"))

    imgs = img.float().unsqueeze(0).cuda()
    print("imgs shape: ", imgs.shape)
    imgs = Variable(imgs)

    outputs = model(imgs)
    print(outputs)
    outputs = F.sigmoid(outputs)
    print(outputs)
    preds = outputs.data.cpu().numpy().tolist()

    return preds[0][0]


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6063)
