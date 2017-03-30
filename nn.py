from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

x = joblib.load('x.pkl')
y = joblib.load('y.pkl')

nn = MLPClassifier(hidden_layer_sizes=(2500,),max_iter=100000,verbose=True,solver='sgd')
mod = nn.fit(x,y)
joblib.dump(mod,'nn_model.pkl')