# RASSAL AĞAÇLARI
# =============================================================================
# Random Forest
# Birden çok parçaya bölerek eğitim yapılır
# Parçalar çok oldukça sonuç daha doğru olacağını amaçlar 
# =============================================================================

import pandas as pd 
import matplotlib.pyplot as plt 


# Veri yükleme
veriler = pd.read_csv('B:\GitHUB\BTK_Akademi_AI\Datasetler/maaslar.csv')

# Data Frame Dilimleme 

x = veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

# Tip Dönüşümü - Numpy Array'e dönüştürüldü
X = x.values
Y = y.values

# Random Forest Ensemble yöntemler içindedir 
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
# n_estimators kaç tane decision çizileceğini gösterir.
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))
print(rf_reg.predict([[11]]))


# plt.scatter(X,Y,color="red")
# plt.plot(X,rf_reg.predict(X),color="blue")

Z = X + 0.5
K= X - 0.5
# plt.plot(X,rf_reg.predict(Z), color="green")
# plt.plot(X,rf_reg.predict(K), color="yellow")

rf_reg2 = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg2.fit(K, Y.ravel())
print(rf_reg2.predict([[6.6]]))
print(rf_reg2.predict([[11]]))














































