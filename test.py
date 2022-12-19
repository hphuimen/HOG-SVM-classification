# -*- coding=utf-8 -*-
import glob
import platform
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
import shutil
import sys
import cv2
from utils import *


def test(test_feat_path, model_path):
    t0 = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0

    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(int(data[-1]))

    clf = joblib.load(model_path + 'model')

    y_pre = []
    for i in range(len(features)):
        feature = features[i].reshape((1, -1)).astype(np.float64)
        result = clf.predict(feature)
        y_pre.append(int(result[0]))
        total += 1
        if int(result[0]) == int(labels[i]):
            correct_number += 1

    acc = float(correct_number) / total
    t1 = time.time()
    print('accuracy in test dataset is： %f' % acc)
    print('spend time : %f' % (t1 - t0))

    class_names = np.array(["cat", "dog"])
    plot_confusion_matrix(labels, y_pre, class_names)
    plt.show()
    precision = precision_test(labels, y_pre)
    recall = recall_test(labels, y_pre)
    f1 = f1_test(labels, y_pre)
    print('precision: %f, recall: %f, f1_score: %f' % (precision, recall, f1))
    plot_roc(labels, y_pre)


test('./feature/test', './model/')
