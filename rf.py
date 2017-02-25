from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

x = joblib.load('x.pkl')
y = joblib.load('y.pkl')

rf = RandomForestClassifier(n_estimators=60)
mod = rf.fit(x,y)
joblib.dump(mod,'rf_model.pkl')