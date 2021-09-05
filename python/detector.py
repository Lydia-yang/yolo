# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
rootpath = os.path.abspath(os.path.join(os.getcwd(), ".."))

import darknet as dn
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#dn.set_gpu(0)

#r = dn.detect(net, meta, "C:/Users/91408/Desktop/yolo/VOCdevkit/VOC2006/PNGImages/000008.png")
#print r
def get_confusion_matrix(net, meta, dataset, model_type):
    predict = []
    true_label = []
    correct = 0
    if dataset == 'VOC2006':
        labels_name = ['bicycle', 'bus', 'car', 'dog', 'cat', 'other']
        dic = {'bicycle':0, 'bus':1, 'car':2, 'dog':3, 'cat':4, 'other':5}
        title = "VOC2006 Confusion Matrix ("+model_type+" Model)"
        filepath = 'data/voc.2006.test'
    else:
        labels_name = ['horse', 'sheep', 'giraffe', 'elephant', 'cow', 'other']
        dic = {'horse': 0, 'sheep': 1, 'giraffe': 2, 'elephant': 3, 'cow': 4, 'other': 5}
        title = "AWA2 Confusion Matrix (" + model_type + " Model)"
        filepath = 'data/awa2.test'
    for line in open(os.path.join(rootpath, filepath),"r"):
        lines = line.strip('\n').split('\t')
        imgpath = lines[0]
        label = lines[-1]
        true_label.append(dic[label])
    #print(imgpath)
        r = dn.detect(net, meta, imgpath)
    #print(r)
        if len(r) >0:
            pre = r[0][0]
        else:
            pre = 'NULL'
    #print(pre)
        if pre not in dic.keys():
            pre = 'other'
          #  print 'other'
        predict.append(dic[pre])
        if pre == label:
            correct += 1
        #break
#print(predict, true_label)
    print("top1 acc:", float(correct)/200)
    cm = confusion_matrix(true_label, predict)
    print(cm)
    plot_confusion_matrix(cm, labels_name, title)
    plt.show()

if __name__ == "__main__":
    model_type = 'small' # 'tiny' or 'small'
    dataset = 'AWA2' # 'VOC2006' or 'AWA2'
    if model_type == 'tiny':
        weights = 'yolov3-tiny.weights'
        cfg = "cfg/yolov3-tiny.cfg"
    else:
        weights = 'yolov2-tiny.weights'
        cfg = "cfg/yolov2-tiny.cfg"
    net = dn.load_net(os.path.join(rootpath, cfg),
                      os.path.join(rootpath,weights), 0)
    meta = dn.load_meta(os.path.join(rootpath, "cfg/coco.data"))
    get_confusion_matrix(net, meta, dataset, model_type)
