# =============================================================================
# ODEV 
# 1. Dataset yüklenimi
# 2. Bağımlı değişkeni ve diğer değişkenlerin ayrımı 
# 3. Kategorik değerlerin sayısal değerlere döndürüp yeni dataset hazırlanması
# 4. Test seti ile dataset ayrımı 
# 5. Model ile tahmin etme 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()


veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/odev_tenis.csv")

veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
c = veriler2.iloc[:,:1] # İlk sütunu alır 
c = ohe.fit_transform(c).toarray()

# DataFrame Hazırlanması
havadurumu = pd.DataFrame(data=c,index=range(14),
                          columns=['overcast','rainy','sunny'])

sonVeriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonVeriler = pd.concat([veriler2.iloc[:,-2:],sonVeriler],axis=1)

# Verilerin Eğitim-Test Kümesi İçin Bölünmesi 
# Humidity Columnu bağımlı, kalanlar bağımsız değişkendir.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(sonVeriler.iloc[:,:-1],
                                                 sonVeriler.iloc[:,-1:],
                                                 test_size=0.33,
                                                 random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)


# backward elimination 

import statsmodels.api as sm
X = np.append(arr=np.ones((14,1)).astype(int),values=sonVeriler.iloc[:,:-1],axis=1)

X_l = sonVeriler.iloc[:,[0,1,2,3,4,5]].values 
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonVeriler.iloc[:,-1:],X_l).fit()
# print(model.summary())

sonVeriler = sonVeriler.iloc[:,1:]

X_l = sonVeriler.iloc[:,[0,1,2,3,4]].values 
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonVeriler.iloc[:,-1:],X_l).fit()
# print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)



