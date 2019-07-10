# fashion_rec_api
This is a set of APIs or tools that are used for filtering the crawled fashion images, recognise the persons, faces, clothes, etc. in the images.

## body_det_darknet
app_6060_get_upload_html_upload_image.py
  An http API to receive the images and htmls uploaded from crawlers, and then push them into storage cluster (here we use cassandra).
app_6061_detect_person.py
  An http API to detect the body bounding boxes of persons within one image. We use darknet whose recognition backbone model is yoloV3 with a pretrained model.

## face_det_mtcnn
app_6062_detect_face.py 
  An http API to detect the faces within one image.

## image_binary_classifier
  Binary image classification model for classifying images into good or bad with regard to whether it is feasible to recognize fashion concepts from those images. We aim to use this classifier to filter out those advertisements, posters, too dark images, blurred images, etc. 

## Other APIs are still under development and testing, coming soon.
