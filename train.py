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

def train(train_feat_path,model_path):
    t0 = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0

    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(data[-1])

    print("Training a Linear LinearSVM Classifier.")
    clf = LinearSVC()
    clf.fit(features, labels)
    # 下面的代码是保存模型的
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(clf, model_path + 'model')

    clf = joblib.load(model_path + 'model')

    for i in range(len(features)):
        feature = features[i].reshape((1, -1)).astype(np.float64)
        result = clf.predict(feature)
        total+=1
        if int(result[0])==int(labels[i]):
            correct_number+=1

    acc=float(correct_number)/total
    t1 = time.time()
    print('accuracy in train dataset is： %f' % acc)
    print('spend time : %f' % (t1 - t0))



train("./feature/train","./model/")

