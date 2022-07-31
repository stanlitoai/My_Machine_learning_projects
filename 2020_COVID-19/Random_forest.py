#loading the library with the miris dataset

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#setting random seed
np.random.seed(0)

#Creating an object called iris with the iris data
iris = load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)

df.head()
#len(df)

df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

df["is_train"] = np.random.uniform(0,1, len(df))<=.75

df.head()







