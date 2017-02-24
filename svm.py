from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

x = joblib.load('x.pkl')
y = joblib.load('y.pkl')

#nn = MLPClassifier(hidden_layer_sizes=(10000,10),activation='logistic',max_iter=100000,tol=0.000001,verbose=True)
sv = SVC(tol=0.0001,verbose=True,decision_function_shape='ovr')
mod = sv.fit(x,y)
joblib.dump(mod,'svm_model.pkl')