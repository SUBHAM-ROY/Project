import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

sift = cv2.SIFT()
arr = np.empty((1,128))
path = "C:\\SUBHAM_MAGIC\\"

# AK47
for i in range(1,10):
    img = cv2.imread(path + "Train\\001.ak47\\001_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,99):
    img = cv2.imread(path + "Train\\001.ak47\\001_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Chess Board
for i in range(1,10):
    img = cv2.imread(path + "Train\\037.chess-board\\037_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,100):
    img = cv2.imread(path + "Train\\037.chess-board\\037_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(100,121):
    img = cv2.imread(path + "Train\\037.chess-board\\037_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Tennis Ball
for i in range(1,10):
    img = cv2.imread(path + "Train\\216.tennis-ball\\216_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,99):
    img = cv2.imread(path + "Train\\216.tennis-ball\\216_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Zebra
for i in range(1,10):
    img = cv2.imread(path + "Train\\250.zebras\\250_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,97):
    img = cv2.imread(path + "Train\\250.zebras\\250_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# CD
for i in range(1,10):
    img = cv2.imread(path + "Train\\033.cds\\033_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,98):
    img = cv2.imread(path + "Train\\033.cds\\033_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Binocular
for i in range(1,10):
    img = cv2.imread(path + "Train\\012.binoculars\\012_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,100):
    img = cv2.imread(path + "Train\\012.binoculars\\012_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(100,197):
    img = cv2.imread(path + "Train\\012.binoculars\\012_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

arr = arr[1:]
print arr.shape
print "Starting Kmeans"

km = KMeans(n_clusters=1000,max_iter=600).fit(arr)
joblib.dump(km,'kmeans.pkl')
print km.labels_