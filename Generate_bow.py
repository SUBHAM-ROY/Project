import cv2
import numpy as np
from sklearn.externals import joblib
import collections

km = joblib.load('kmeans.pkl')

sift = cv2.SIFT()
x = np.empty((1,1000))
y = np.empty(1)

for i in range(1,10):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Train\\001.ak47\\001_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[1]))

for i in range(10,99):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Train\\001.ak47\\001_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[1]))
    
for i in range(1,10):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Train\\037.chess-board\\037_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[2]))

for i in range(10,100):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Train\\037.chess-board\\037_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[2]))

for i in range(100,121):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Train\\037.chess-board\\037_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[2]))

for i in range(1,10):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Train\\216.tennis-ball\\216_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[3]))
    
for i in range(10,99):
    img = cv2.imread("D:\\Sem_Releated\\8th Sem\\Project\\Train\\216.tennis-ball\\216_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[3]))
    
x = x[1:]
y = y[1:]

joblib.dump(x,'x.pkl')
joblib.dump(y,'y.pkl')