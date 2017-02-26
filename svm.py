from sklearn.externals import joblib
from sklearn.svm import SVC

x = joblib.load('x.pkl')
y = joblib.load('y.pkl')

sv = SVC(tol=0.00001,kernel='linear',decision_function_shape='ovr')
mod = sv.fit(x,y)
joblib.dump(mod,'svm_model.pkl')