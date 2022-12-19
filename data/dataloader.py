import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import joblib
from skimage.feature import hog


dict = {'cat': 0, 'dog': 1}
def getfiles(file):
    path_list = []
    labels = []
    filenames = os.listdir(file)
    for filename in filenames:
        a = os.path.join(file, filename)
        label = dict[filename.split('.')[0]]
        labels.append(label)
        path_list.append(a)
    return path_list,labels


def get_image_list(piclist, w, h):
    img_list = []
    for pic in piclist:
        temp = Image.open(pic)
        temp = temp.resize((w, h), Image.ANTIALIAS)
        img_list.append(temp.copy())
        temp.close()
    return img_list


def img_shuff(imglist, h, w, labels):
    N = len(imglist)
    N_train = N // 5 * 4

    imgs = np.zeros([N, h, w, 3], dtype=np.uint8)
    for i in range(len(imglist)):
        imgs[i] = np.reshape(imglist[i], (h, w, -1))

    index = [i for i in range(8000)]
    random.shuffle(index)
    labels = labels[index]
    imgs = imgs[index]

    return imgs[0:N_train], labels[0:N_train], imgs[N_train:], labels[N_train:]

def get_feat(imgs, labels, savePath):
    N = len(imgs)
    for i in range(N):
        img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)

        fd , hog_image= hog(img, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[4, 4],
                            visualize=True,transform_sqrt=True)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('hog_image', hog_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fd = np.concatenate((fd, [labels[i]]))
        fd_name = 'hogfuture'+ str(i) + '.feat'
        fd_path = os.path.join(savePath, fd_name)
        joblib.dump(fd, fd_path)
    print("features are extracted and saved.")



def extra_feat(file,w,h):
    piclist, labels = getfiles(file)
    labels=np.array(labels)
    imgs = get_image_list(piclist,w,h)
    train_imgs,train_labels,test_imgs, test_labels=img_shuff(imgs,h,w,labels)
    get_feat(train_imgs, train_labels, '../feature/train/')
    get_feat(test_imgs, test_labels, '../feature/test/')

extra_feat('../cat_dog/train',64,64)

