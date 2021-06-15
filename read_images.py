import cv2
import numpy as np
import os

# 0,1,2
artists = ['Vincent_van_Gogh', 'Pablo_Picasso', 'Edgar_Degas']
X = []
y = []

for artist in artists:
    for image in os.listdir(os.path.join("data", artist)):
        img_normal = cv2.imread('data/' + artist + '/' + image)
        img_resized = cv2.resize(img_normal, (512,512))
        X.append(img_resized)
        if artist == 'Vincent_van_Gogh':
            y.append(0)
        elif artist == 'Pablo_Picasso':
            y.append(1)
        elif artist == 'Edgar_Degas':
            y.append(2)
        # elif artist == 'Vincent_van_Gogh':
        #    y.append(3)

_X = np.array(X)
_y = np.array(y)

np.save('data.npy', _X)
np.save('labels.npy', _y)
