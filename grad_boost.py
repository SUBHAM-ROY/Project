from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

x = joblib.load('x.pkl')
y = joblib.load('y.pkl')

gb = GradientBoostingClassifier(n_estimators=500,max_features=None)
mod = gb.fit(x,y)
joblib.dump(mod,'gb_model.pkl')