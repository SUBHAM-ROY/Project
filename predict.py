import cv2
import collections
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

sift = cv2.SIFT()
km = joblib.load('kmeans.pkl')
mod = joblib.load('svm_model.pkl')

y_true = np.concatenate((np.ones(20),np.ones(20)*2,np.ones(20)*3))
y_pred = [1]

for i in range(1,21):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Test\\ak47\\"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Test\\chess\\"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Test\\tennis\\"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))
#    if(lab==[1]):
#        print "AK 47 "+str(i)
#    elif(lab==[2]):
#        print "Chess Board "+str(i)
#    elif(lab==[3]):
#        print "Tennis Ball "+str(i)
#    else:
#        print "Not recognized"

y_pred = y_pred[1:]

scr = precision_recall_fscore_support(y_true,y_pred,average='micro')
print scr