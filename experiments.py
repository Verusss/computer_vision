import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

X = np.load('data_lbp.npy', allow_pickle = True)
y = np.load('labels.npy')

svc = SVC()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

accuracy = balanced_accuracy_score(y_test, y_pred)

print(accuracy)
