from sklearn.datasets import load_boston
b_dataset = load_boston()


import pandas as pd


df_features = pd.DataFrame(b_dataset.data, columns = b_dataset.feature_names)
df_target = pd.DataFrame(b_dataset.target, columns = ["prices"])
df = pd.concat([df_features, df_target], axis =1)

from sklearn.model_selection    import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score
df.corr()
df.shape
df.dropna()
df.describe()

X= df_features
y=df_target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0 )
classifier = RandomForestRegressor(n_estimators = 10, random_state = 0)
classifier.fit(X_train, y_train)

pred_rfc = classifier.predict(X_test)


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize= (12, 10))
sns.heatmap(df.corr(), annot =True)







y_pred  = learn _model.predict(X_test)

df_pred_actual = pd.DataFrame({"predicted": y_pred, "actual": y_test})
df_pred_actual.head()

#for checking your score
from sklearn.metrics import r2_score

print("Testing_score: ", r2_score(y_test, pred_rfc)*100)





















