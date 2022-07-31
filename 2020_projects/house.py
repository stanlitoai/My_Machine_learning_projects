import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection    import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

data= pd.read_csv("HousePrices.csv")

X = data.iloc[:, : -1].values
y = data.iloc[:, 15].values

pd.isnull(data)
data.shape



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#X_train, X_test, y_train, y_test = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
                        

rec = RandomForestRegressor(n_estimators = 10, random_state = 0)

rec.fit(X_train, y_train)

pred = rec.predict(X_test)

pred

from sklearn.metrics import r2_score

print("Testing_score: ", r2_score(y_test, pred)*100)

df_pred_actual = pd.DataFrame({"predicted": pred, "actual": y_test})
df_pred_actual.head()

#if pred >= 90:
 #   print("congrats!!!")
#else:
 #   print("sorry! train your model again")    
scaler = MinMaxScaler(feature_range = (0,1))
x = scaler.fit_transform(X)


scores = []
best_svr = SVR(kernel= "rbf")
cv = KFold(n_splits = 10, random_state = 42, shuffle= False)

for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index, "\n")
    
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_svr.fit(X_train , y_train)
    
scores.append(best_svr(X_test, y_test))





















