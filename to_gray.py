import numpy as np
import cv2

X_color = np.load('data_npy/data.npy', allow_pickle = True)

X_gray = []
for image in X_color:
    X_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

X = np.array(X_gray)

np.save('data_npy/data_gray.npy', X)
