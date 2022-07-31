from sklearn import svm
from sklearn import datasets
#import Seaborn as sns
from sklearn.model_selection    import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score




Iris = datasets.load_iris()
type(Iris)

Iris.data
Iris.target
Iris.feature_names

Iris.target_names

X = Iris.data
y = Iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(x_train, y_train)
pred_rfc = rfc.predict(x_test)

a =classification_report(y_test, pred_rfc)
b = confusion_matrix(y_test, pred_rfc)
s = accuracy_score(y_test, pred_rfc)
a