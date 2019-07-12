#!/usr/bin/env python3

import cv2
import json
import requests
import argparse
import numpy as np


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main():    
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image', default='./test.jpg', help='path of the image to test')
    parser.add_argument('--ret_img', default=False, type=bool, help='whether to return the labelled image')
    parser.add_argument('--url', default="http://gpu1:6064/age_gender_estimation", help='the http url')
    args = parser.parse_args()
    
    url = args.url 
    files = {"image": open(args.image, 'rb')}
    
    res = requests.post(url, files=files).json()
    print(json.dumps(res, indent=4))
    
    if res["T_F"] is True:
        img = cv2.imread(args.image)
        res = res['result']
        bbox = np.array(res['bbox'])
        points = np.array(res['points'])
        age = np.array(res['age'])
        gender = np.array(res['gender'])

        tmp = args.image.split("/")
        tmp1 = tmp[1].split(".")
        new_img_path = tmp[0] + "/" + tmp1[0] + "_result." + tmp1[1] 
        print(new_img_path)

        for b in bbox:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        for p in points:
            for i in range(5):
                cv2.circle(img, (p[i][0], p[i][1]), 1, (0, 0, 255), 2)
        for i in range(len(age)):
            label = "{}, {}".format(int(age[i]), "F" if gender[i] == 0 else "M")
            draw_label(img, (int(bbox[i,0]), int(bbox[i,1])), label)

        cv2.imwrite(new_img_path, img)


if __name__ == "__main__":
    main()
