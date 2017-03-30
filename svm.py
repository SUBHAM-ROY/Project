from sklearn.externals import joblib
from sklearn.svm import SVC

x = joblib.load('x.pkl')
y = joblib.load('y.pkl')

sv = SVC(tol=0.001,kernel='poly',degree=2,decision_function_shape='ovr')
mod = sv.fit(x,y)
joblib.dump(mod,'svm_model.pkl')