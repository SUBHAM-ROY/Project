import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

sift = cv2.SIFT()
arr = np.empty((1,128))
path = "./"

# AK47
for i in range(1,10):
    img = cv2.imread(path + "Train/001.ak47/001_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,99):
    img = cv2.imread(path + "Train/001.ak47/001_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Chess Board
for i in range(1,10):
    img = cv2.imread(path + "Train/037.chess-board/037_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,100):
    img = cv2.imread(path + "Train/037.chess-board/037_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(100,121):
    img = cv2.imread(path + "Train/037.chess-board/037_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Tennis Ball
for i in range(1,10):
    img = cv2.imread(path + "Train/216.tennis-ball/216_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,99):
    img = cv2.imread(path + "Train/216.tennis-ball/216_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Zebra
for i in range(1,10):
    img = cv2.imread(path + "Train/250.zebras/250_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,97):
    img = cv2.imread(path + "Train/250.zebras/250_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# CD
for i in range(1,10):
    img = cv2.imread(path + "Train/033.cds/033_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,98):
    img = cv2.imread(path + "Train/033.cds/033_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Binocular
for i in range(1,10):
    img = cv2.imread(path + "Train/012.binoculars/012_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,100):
    img = cv2.imread(path + "Train/012.binoculars/012_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(100,197):
    img = cv2.imread(path + "Train/012.binoculars/012_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Hibiscus
for i in range(1,10):
    img = cv2.imread(path + "Train/103.hibiscus/103_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,100):
    img = cv2.imread(path + "Train/103.hibiscus/103_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(100,112):
    img = cv2.imread(path + "Train/103.hibiscus/103_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Ketch
for i in range(1,10):
    img = cv2.imread(path + "Train/123.ketch/123_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,100):
    img = cv2.imread(path + "Train/123.ketch/123_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(100,112):
    img = cv2.imread(path + "Train/123.ketch/123_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Minaret
for i in range(1,10):
    img = cv2.imread(path + "Train/143.minaret/143_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,100):
    img = cv2.imread(path + "Train/143.minaret/143_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(100,131):
    img = cv2.imread(path + "Train/143.minaret/143_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

# Sneaker
for i in range(1,10):
    img = cv2.imread(path + "Train/191.sneaker/191_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(10,100):
    img = cv2.imread(path + "Train/191.sneaker/191_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

for i in range(100,112):
    img = cv2.imread(path + "Train/191.sneaker/191_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    arr = np.concatenate((arr,desc))

arr = arr[1:]
print arr.shape
print "Starting Kmeans"

km = KMeans(n_clusters=1000,n_init=1,precompute_distances=False,n_jobs=1,algorithm="full").fit(arr)
joblib.dump(km,'kmeans.pkl')
print km.labels_