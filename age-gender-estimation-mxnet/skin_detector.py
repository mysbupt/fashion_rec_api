from sklearn.cluster import KMeans
from scipy import spatial
import cv2
import numpy as np

# This function is meant to give the skin color of the person by detecting face and then
# applying k-Means Clustering.

def get_rgb_name_file(rgb_file,name_file):
    name_dict = {}
    rgb_list = []
    f_name = open(name_file).readlines()
    f_rgb = open(rgb_file).readlines()
    for record in f_name:
        re1 = record.rstrip('\n')
        name = re1.split(',')
        name_dict[name[0]] = name[1]
    for record in f_rgb:
        re2 = record.rstrip('\n').split(',')
        rgb = []
        for i in re2:
            rgb.append(int(i))
        rgb_list.append(rgb)
    return rgb_list,name_dict

def get_color_name(RGB,HexNameDict,img):
    NearestRGB= RGB[spatial.KDTree(RGB).query(img)[1]]
    ColorHex = format(NearestRGB[0],'x').zfill(2)\
            + format(NearestRGB[1],'x').zfill(2)\
            + format(NearestRGB[2],'x').zfill(2)
    ColorDiff = \
            '(' + '{0:+d}'.format(NearestRGB[0]-img[0])\
            + ',' + '{0:+d}'.format(NearestRGB[1]-img[1])\
            + ',' + '{0:+d}'.format(NearestRGB[2]-img[2])\
            +')'
    try:
        ColorName = HexNameDict[ColorHex]
    except:
        ColorName = 'not found'
    return ColorName

def get_skin_color(image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters = 4)
        clt.fit(image)

        def centroid_histogram(clt):
                # Grab the number of different clusters and create a histogram
            # based on the number of pixels assigned to each cluster.
            numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
            (hist, _) = np.histogram(clt.labels_, bins = numLabels)

            # Normalize the histogram, such that it sums to one.
            hist = hist.astype("float")
            hist /= hist.sum()

            # Return the histogram.
            return hist

        def get_color(hist, centroids):

            # Obtain the color with maximum percentage of area covered.
            maxi=0
            COLOR=[0,0,0]

            # Loop over the percentage of each cluster and the color of
            # each cluster.
            for (percent, color) in zip(hist, centroids):
                if(percent>maxi):
                    COLOR=color
                    maxi=percent
            # Return the most dominant color.
            return COLOR

        # Obtain the color and convert it to HSV type
        hist = centroid_histogram(clt)
        skin_temp1 = get_color(hist, clt.cluster_centers_)
        skin_color = np.uint8([[skin_temp1]])
        skin_color = skin_color[0][0]
        # Return the color.
        return skin_color

