import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib

from sklearn.model_selection import train_test_split, KFold


data = pd.read_csv("Weber_plc.csv")

data.info()

pd.isnull(data)
X = data.iloc[:,: -1].values
y = data.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(handle_unknown='ignore' )
X = onehotencoder.fit_transform(X).toarray()

from sklearn.ensemble import RandomForestClassifier

rd = RandomForestClassifier(n_estimators = 200)
rd.fit(X, y)
pred= rd.predict(X)


from sklearn.metrics import accuracy_score
m
accuracy_score(y, pred)*100

data.corr()


plt.scatter(y, pred)
plt.title("Weber plc", fontsize =  20)
plt.xlabel("X axis", fontsize =  15)
plt.ylabel("y axis", fontsize =  15)
plt.show()



















#another work








import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#data = pd.read_csv("people.csv")
X = np.arange(10).reshape((5,2))
y =  range(5)

y

from sklearn.model_selection import train_test_split, KFold


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle= False, random_state = None)


kf = KFold(n_splits= 4, shuffle = False).split(range(16))














