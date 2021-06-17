import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog

numPoints = 12
radius = 8
X = np.load('data_gray_test.npy', allow_pickle = True)

X_lbp = []
for image in X:
    lbp_image = local_binary_pattern(image, numPoints, radius)
    #fd, hog_image = hog(lbp_image, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize =True)
    X_lbp.append(hog(lbp_image, pixels_per_cell=(4, 4), cells_per_block=(1, 1)))

#cv2.imwrite('test.jpg', lbp_image)
#cv2.imwrite('test1.jpg', hog_image)

np.save('third_version/data_lbp_12_8_44_11.npy', X_lbp)
