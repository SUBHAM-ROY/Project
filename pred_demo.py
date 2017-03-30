import cv2
import collections
from sklearn.externals import joblib

sift = cv2.SIFT()
km = joblib.load('kmeans.pkl')
mod = joblib.load('nn_model.pkl')
path = "./"

img = cv2.imread(path + "Demo/2.jpg",0)
pts, desc = sift.detectAndCompute(img,None)
cnt = collections.Counter(km.predict(desc))
lst = [cnt[j] for j in range(0,1000)]
lab = mod.predict([lst])

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
elif(lab==[7]):
    print "Hibiscus "
elif(lab==[8]):
    print "Ketch "
elif(lab==[9]):
    print "Minaret "
elif(lab==[10]):
    print "Sneaker "
