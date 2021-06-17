import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import pickle

datasets_number_of_points = ['data_lbp_8_4_44_11.npy', 'data_lbp_12_4_44_11.npy', 'data_lbp_16_4_44_11.npy']
datasets_radius = ['data_lbp_8_4_44_11.npy', 'data_lbp_8_8_44_11.npy', 'data_lbp_8_12_44_11.npy', 'data_lbp_8_16_44_11.npy']


Xs_number_of_points = []
for dataset in datasets_number_of_points:
     Xs_number_of_points.append(np.load('data_npy/' + dataset, allow_pickle = True))

Xs_radius = []
for dataset in datasets_radius:
     Xs_radius.append(np.load('data_npy/' + dataset, allow_pickle = True))

y = np.load('data_npy/labels.npy')

radius = [4, 8, 12, 16]
number_of_points = [8, 12, 16]

accuracies_svc_number_of_points = []
accuracies_gnb_number_of_points = []
for i, X in enumerate(Xs_number_of_points):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    accuracies_svc_number_of_points.append(balanced_accuracy_score(y_test, y_pred))
    filename = 'models/svc_number_of_points_' + str(number_of_points[i]) + '.sav'
    pickle.dump(svc, open(filename, 'wb'))

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracies_gnb_number_of_points.append(balanced_accuracy_score(y_test, y_pred))
    filename = 'models/gnb_number_of_points_' + str(number_of_points[i]) + '.sav'
    pickle.dump(svc, open(filename, 'wb'))

accuracies_svc_radius = []
accuracies_gnb_radius = []
for i, X in enumerate(Xs_radius):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    accuracies_svc_radius.append(balanced_accuracy_score(y_test, y_pred))
    filename = 'models/svc_radius_' + str(radius[i]) + '.sav'
    pickle.dump(svc, open(filename, 'wb'))

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracies_gnb_radius.append(balanced_accuracy_score(y_test, y_pred))
    filename = 'models/gnb_radius_' + str(radius[i]) + '.sav'
    pickle.dump(svc, open(filename, 'wb'))


print("NUMBER OF POINTS")
print("SVC")
print(accuracies_svc_number_of_points)

print("GNB")
print(accuracies_gnb_number_of_points)

plt.figure()
plt.title('SVC')
plt.xlabel('Number of points')
plt.ylabel('Balanced accuracy score')
plt.plot(number_of_points, accuracies_svc_number_of_points, marker = 'o')
plt.savefig('figures/accuracy_number_of_points_svc.png')

plt.figure()
plt.title('GNB')
plt.xlabel('Number of points')
plt.ylabel('Balanced accuracy score')
plt.plot(number_of_points, accuracies_gnb_number_of_points, marker = 'o')
plt.savefig('figures/accuracy_number_of_points_gnb.png')

plt.figure()
plt.title('SVC and GNB')
plt.xlabel('Number of points')
plt.ylabel('Balanced accuracy score')
plt.plot(number_of_points, accuracies_svc_number_of_points, 'r', number_of_points, accuracies_gnb_number_of_points, 'g')
plt.legend(['SVC', 'GNB'])
plt.savefig('figures/accuracy_number_of_points_svc_gnb.png')

print("RADIUS")
print("SVC")
print(accuracies_svc_radius)

print("GNB")
print(accuracies_gnb_radius)

plt.figure()
plt.title('SVC')
plt.xlabel('Radius')
plt.ylabel('Balanced accuracy score')
plt.plot(radius, accuracies_svc_radius, marker = 'o')
plt.savefig('figures/accuracy_radius_svc.png')

plt.figure()
plt.title('GNB')
plt.xlabel('Radius')
plt.ylabel('Balanced accuracy score')
plt.plot(radius, accuracies_gnb_radius, marker = 'o')
plt.savefig('figures/accuracy_radius_gnb.png')

plt.figure()
plt.title('SVC and GNB')
plt.xlabel('Radius')
plt.ylabel('Balanced accuracy score')
plt.plot(radius, accuracies_svc_radius, 'r', radius, accuracies_gnb_radius, 'g')
plt.legend(['SVC', 'GNB'])
plt.savefig('figures/accuracy_radius_svc_gnb.png')
