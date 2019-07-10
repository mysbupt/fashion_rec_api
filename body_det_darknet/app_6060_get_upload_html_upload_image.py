#!/usr/bin/env python

import os
from flask import Flask, request, json
#from darknet import *
import tempfile
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import SimpleStatement

import yaml

conf = yaml.load(open("./config.yaml"))

app = Flask(__name__)


#net = load_net("./cfg/yolov3.cfg", "./models/yolov3.weights", 0)
#meta = load_meta("./cfg/coco.data")


hostname = conf["cassandra"]["host"]
keyspace_ltk = 'liketoknowit'
keyspace_ins = conf["cassandra"]["keyspace"] 
nodes = [hostname]
cluster = Cluster(nodes)
session_ltk = cluster.connect(keyspace_ltk)
session_ins = cluster.connect(keyspace_ins)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    res = {
        'msg': 'success',
    }

    if 'image' not in request.files:
        res['msg'] = 'error: no image uploaded'
        response = app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )
        return response
    else:
        image = request.files['image']

    try:
        database = request.form['database']
    except:
        res['msg'] = 'error: you need specify "database" parameter in form'
        return app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )
 
    if database not in ('liketoknowit', 'instagram_scene'):
        res['msg'] = 'error: invalid value of "database"'
        return app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )
    
    tmp_file, image_tmp_path = tempfile.mkstemp()
    image.save(image_tmp_path)
    img_url_md5 = image.filename.split('/')[-1].split('.')[0]

    CQL_str = "INSERT INTO images (img_url_md5, image) VALUES (?, ?)"
    img_data = open(image_tmp_path).read()
    if database == 'liketoknowit':
        p = session_ltk.prepare(CQL_str)
        session_ltk.execute(p, [img_url_md5, img_data])
    else:
        p = session_ins.prepare(CQL_str)
        session_ins.execute(p, [img_url_md5, img_data])
    print(img_url_md5, ': successfully save to cassandra', res)
    os.close(tmp_file)
    os.remove(image_tmp_path)

    return app.response_class(
        response = json.dumps(res),
        status = 200,
        mimetype = 'application/json'
    )


@app.route('/upload_html', methods=['POST'])
def upload_html():
    res = {
        'msg': 'success',
    }

    req = request.get_json()
    if 'html_url_md5' not in req.keys() or 'html' not in req.keys() or 'database' not in req.keys():
        res['msg'] = 'error: you need include "html_url_md5", "html", and "database" in your request'
        return app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )
    else:
        html_url_md5 = req['html_url_md5']
        html_data = req['html']
        database = req['database']

    if database not in ('liketoknowit', 'instagram_scene'):
        res['msg'] = 'error: invalid value of "database"'
        return app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )
    

    CQL_str = "INSERT INTO htmls (html_url_md5, html) VALUES (?, ?)"
    if database == 'liketoknowit':
        p = session_ltk.prepare(CQL_str)
        session_ltk.execute(p, [html_url_md5, html_data])
    else:
        p = session_ins.prepare(CQL_str)
        session_ins.execute(p, [html_url_md5, html_data])
    print(html_url_md5, ': successfully save to cassandra', res)

    return app.response_class(
        response = json.dumps(res),
        status = 200,
        mimetype = 'application/json'
    )


@app.route('/get_html', methods=['POST'])
def get_html():
    res = {
        'msg': 'success',
    }

    req = request.get_json()
    print req
    if 'html_url_md5' not in req.keys():
        res['msg'] = 'error: you need include "html_url_md5" in your request'
        return app.response_class(
            response = json.dumps(res),
            status = 200,
            mimetype = 'application/json'
        )
    else:
        html_url_md5 = req['html_url_md5']

    CQL_str = "SELECT html from htmls WHERE html_url_md5 = ?"
    p = session_ins.prepare(CQL_str)
    res['html'] = session_ins.execute(p, [html_url_md5])[0]
    print(html_url_md5, ': successfully get html')

    return app.response_class(
        response = json.dumps(res),
        status = 200,
        mimetype = 'application/json'
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6060)
