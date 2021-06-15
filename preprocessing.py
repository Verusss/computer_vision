import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog

numPoints = 8 #12 #8
radius = 4 #8 #12
X = np.load('data_gray.npy', allow_pickle = True)

X_lbp = []
for image in X:
    lbp_image = local_binary_pattern(image, numPoints, radius)
    #fd, hog_image = hog(lbp_image, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize =True)
    X_lbp.append(hog(lbp_image, pixels_per_cell=(4, 4), cells_per_block=(1, 1)))

#cv2.imwrite('test.jpg', lbp_image)
#cv2.imwrite('test1.jpg', hog_image)

_X = np.array(X_lbp)

np.save('data_lbp.npy', _X)
