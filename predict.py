import cv2
import collections
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

sift = cv2.SIFT()
km = joblib.load('kmeans.pkl')
mod = joblib.load('svm_model.pkl')
path = "./"
y_true = np.concatenate((np.ones(20),np.ones(20)*2,np.ones(20)*3))
y_true = np.concatenate((y_true,np.ones(20)*4,np.ones(20)*5,np.ones(20)*6))
y_true = np.concatenate((y_true,np.ones(20)*7,np.ones(20)*8,np.ones(20)*9,np.ones(20)*10))
y_pred = [1]

for i in range(1,21):
    img = cv2.imread(path + "Test/ak47/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/chess/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/tennis/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/zebra/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/cd/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/bino/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/hibiscus/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/ketch/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/minaret/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

for i in range(1,21):
    img = cv2.imread(path + "Test/sneakers/"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[j] for j in range(0,1000)]
    lab = mod.predict([lst])
    y_pred = np.concatenate((y_pred,lab))

y_pred = y_pred[1:]

scr = precision_recall_fscore_support(y_true,y_pred,average='micro')
print scr