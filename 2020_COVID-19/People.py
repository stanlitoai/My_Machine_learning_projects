import pandas as pd
import tensorflow

from sklearn.datasets import load_breast_cancer

lbc = load_breast_cancer()

df_features = pd.DataFrame(lbc.data, columns = lbc.feature_names)
df_target = pd.DataFrame(lbc.target, columns = ["canc"])

df = pd.concat([df_features, df_target], axis =1)


df.head()

print(lbc.DESCR)