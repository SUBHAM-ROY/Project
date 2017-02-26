from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB

x = joblib.load('x.pkl')
y = joblib.load('y.pkl')

nb = MultinomialNB()
mod = nb.fit(x,y)
joblib.dump(mod,'nb_model.pkl')