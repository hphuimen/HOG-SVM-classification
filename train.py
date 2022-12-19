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
import seaborn
import matplotlib.pyplot as plt

from utils import precision_test, recall_test, f1_test


def train(train_feat_path,model_path):
    t0 = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0

    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(int(data[-1]))

    '''print("Training res Linear LinearSVM Classifier.")
    clf = LinearSVC()
    clf.fit(features, labels)
    # 下面的代码是保存模型的
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(clf, model_path + 'model')'''

    clf = joblib.load(model_path + 'model')
    res = np.zeros((2, 2))

    y_pre = []
    for i in range(len(features)):
        feature = features[i].reshape((1, -1)).astype(np.float64)
        result = clf.predict(feature)
        y_pre.append(int(result[0]))
        total+=1
        if int(result[0])==1 and int(labels[i])==1:
            res[1][1]+=1
        elif int(result[0])==1 and int(labels[i])==0:
            res[1][0]+=1
        elif int(result[0])==0 and int(labels[i])==1:
            res[0][1]+=1
        else:
            res[0][0]+=1


    acc= float(res[1][1] + res[0][0]) / total
    t1 = time.time()
    print('accuracy in train dataset is： %f' % acc)
    print('spend time : %f' % (t1 - t0))
    res=res/total
    print(res)
    seaborn.heatmap(res, cmap="YlGnBu", annot=True)
    plt.show()
    precision = precision_test(labels, y_pre)
    recall = recall_test(labels, y_pre)
    f1 = f1_test(labels, y_pre)
    print('precision: %f, recall: %f, f1_score: %f' % (precision, recall, f1))



train("./feature/train","./model/")

