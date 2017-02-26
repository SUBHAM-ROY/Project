import cv2
import collections
from sklearn.externals import joblib
import numpy as np

sift = cv2.SIFT()
km = joblib.load('kmeans.pkl')
mod = joblib.load('nn_model.pkl')
path = "./"
y_true = np.concatenate((np.ones(20),np.ones(20)*2,np.ones(20)*3))
y_true = np.concatenate((y_true,np.ones(20)*4,np.ones(20)*5,np.ones(20)*6))
y_pred = [1]


img = cv2.imread(path + "Demo/6.jpg",0)
pts, desc = sift.detectAndCompute(img,None)
cnt = collections.Counter(km.predict(desc))
lst = [cnt[j] for j in range(0,1000)]
lab = mod.predict([lst])
y_pred = np.concatenate((y_pred,lab))
if(lab==[1]):
    print "AK 47 "
elif(lab==[2]):
    print "Chess Board "
elif(lab==[3]):
    print "Tennis Ball "
elif(lab==[4]):
    print "Zebra "
elif(lab==[5]):
    print "CD "
elif(lab==[6]):
    print "Binocular "
