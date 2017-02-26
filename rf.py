from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

x = joblib.load('x.pkl')
y = joblib.load('y.pkl')

rf = RandomForestClassifier(n_estimators=155,max_features=None)
mod = rf.fit(x,y)
joblib.dump(mod,'rf_model.pkl')