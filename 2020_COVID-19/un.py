from sklearn.linear_model import LinearRegression 
import pandas as pd
import numpy as np

import cv2

bigcity = pd.read_csv("bigcity.csv")
orders = pd.read_csv("orders.csv")
samplesub = pd.read_csv("Samplesubmission.csv")
test_customers = pd.read_csv("test_customers.csv")
test_locations = pd.read_csv("test_locations.csv")
train_customers = pd.read_csv("train_customers.csv")
train_locations = pd.read_csv("train_locations.csv")
vendors = pd.read_csv("vendors.csv")

#df = pd.DataFrame("bigcity","orders","vendors","samplesub")

merge = bigcity.merge(orders)
merge.to_csv("orders.csv", index=False)

print(merge.head)