import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data= pd.read_excel('Rain_days.xlsx')

data.head()

a=data['date'] = data['Date and Time'].apply(lambda x: x.split('at')[0])
b=data['time'] = data['Date and Time'].apply(lambda x: x.split('at')[1])




data['Datetime'] = data['Date and Time'].apply(lambda x: ''.join(x.split('at')))

#data.Datetime.dtype

data['Datetime'] = pd.to_datetime(data.Datetime)

#data.Datetime.dtype



c=data['Temp C'] = data['Temp C'].str.replace(r'C', '')
d=data['Temp F'] = data['Temp F'].str.replace(r'F', '')

#converting to numeric
e=data['Temp C'] = data['Temp C'].astype(float)
f=data['Temp F'] = data['Temp F'].astype(float)


g=data.Type.unique()

g=data['year'] = data.Datetime.dt.year
h=data['month'] = data.Datetime.dt.month
i=data['day'] = data.Datetime.dt.day
data['dayofweek'] = data.Datetime.dt.dayofweek
o=data['hour'] = data.Datetime.dt.hour
p=data['minute'] = data.Datetime.dt.minute
l=data['week'] = data.Datetime.dt.week
t=data['weather'] = data.iloc[:,1]

a =data.head()


data.drop(data['dayofweek'])

df = pd.DataFrame({"Month": h, "Days":i, "Years": g, "Hours": o, "Minutes": p,"Temp F": f, "Temp C": e, "Weather": t})


X= df.iloc[:,: -1].values
y = df.iloc[:, 7].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

onehotencoder = OneHotEncoder(handle_unknown='ignore' )
y = onehotencoder.fit_transform(y).toarray().reshape(1,-1)





















data_time = pd.DataFrame({"Date": data['date'], "Time": data['time']})




