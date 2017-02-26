import cv2
import numpy as np
from sklearn.externals import joblib
import collections

km = joblib.load('kmeans.pkl')
path = "./"
sift = cv2.SIFT()
x = np.empty((1,1000))
y = np.empty(1)

# AK47
for i in range(1,10):
    img = cv2.imread(path + "Train/001.ak47/001_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[1]))

for i in range(10,99):
    img = cv2.imread(path + "Train/001.ak47/001_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[1]))
    
# Chess Board
for i in range(1,10):
    img = cv2.imread(path + "Train/037.chess-board/037_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[2]))

for i in range(10,100):
    img = cv2.imread(path + "Train/037.chess-board/037_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[2]))

for i in range(100,121):
    img = cv2.imread(path + "Train/037.chess-board/037_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[2]))

# Tennis Ball
for i in range(1,10):
    img = cv2.imread(path + "Train/216.tennis-ball/216_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[3]))
    
for i in range(10,99):
    img = cv2.imread(path + "Train/216.tennis-ball/216_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[3]))

# Zebra
for i in range(1,10):
    img = cv2.imread(path + "Train/250.zebras/250_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[4]))
    
for i in range(10,97):
    img = cv2.imread(path + "Train/250.zebras/250_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[4]))

# CD
for i in range(1,10):
    img = cv2.imread(path + "Train/033.cds/033_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[5]))
    
for i in range(10,98):
    img = cv2.imread(path + "Train/033.cds/033_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[5]))

# Binoculars
for i in range(1,10):
    img = cv2.imread(path + "Train/012.binoculars/012_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[6]))

for i in range(10,100):
    img = cv2.imread(path + "Train/012.binoculars/012_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[6]))

for i in range(100,197):
    img = cv2.imread(path + "Train/012.binoculars/012_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[6]))

# Hibiscus
for i in range(1,10):
    img = cv2.imread(path + "Train/103.hibiscus/103_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[7]))

for i in range(10,100):
    img = cv2.imread(path + "Train/103.hibiscus/103_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[7]))

for i in range(100,112):
    img = cv2.imread(path + "Train/103.hibiscus/103_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[7]))

# Ketch
for i in range(1,10):
    img = cv2.imread(path + "Train/123.ketch/123_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[8]))

for i in range(10,100):
    img = cv2.imread(path + "Train/123.ketch/123_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[8]))

for i in range(100,112):
    img = cv2.imread(path + "Train/123.ketch/123_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[8]))

# Minaret
for i in range(1,10):
    img = cv2.imread(path + "Train/143.minaret/143_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[9]))

for i in range(10,100):
    img = cv2.imread(path + "Train/143.minaret/143_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[9]))

for i in range(100,131):
    img = cv2.imread(path + "Train/143.minaret/143_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[9]))

# Sneaker
for i in range(1,10):
    img = cv2.imread(path + "Train/191.sneaker/191_000"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[10]))

for i in range(10,100):
    img = cv2.imread(path + "Train/191.sneaker/191_00"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[10]))

for i in range(100,112):
    img = cv2.imread(path + "Train/191.sneaker/191_0"+str(i)+".jpg",0)
    pts, desc = sift.detectAndCompute(img,None)
    cnt = collections.Counter(km.predict(desc))
    lst = [cnt[i] for i in range(0,1000)]
    x = np.concatenate((x,[lst]))
    y = np.concatenate((y,[10]))

x = x[1:]
y = y[1:]

joblib.dump(x,'x.pkl')
joblib.dump(y,'y.pkl')