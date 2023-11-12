# =============================================================================
# DEĞERLENDİRME ve METOTLARIN KARŞILAŞTIRILMASI
# =============================================================================

# R2 Hesaplama
# Hata kareler toplamı ile
# Hata Kare = (tahmin-gerçek)^2 
# Tahminden ne kadar saptık = (gerçek-tahminORT)^2
# R2 = 1 - HataKare/TahminSapması
# R2 değeri 0 => Dünyanın en kötü Algorştmasıdır. Ortalamayı döndürür
# R2 değeri - değer => En aptal yapay zekadan da aptal 
# Amaç = R2 değeri 1e yakın olmdurmaya çalışıyoruz


# =============================================================================
# Düzeltilmiş R2 Hesaplama - Adjusted R2 
# =============================================================================
"""
               (n-1)
R2 = 1-(1-R^2) ______
               n-p-1
r2 de yetersiz kalabilir. 
"""


import pandas as pd 


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

from sklearn.metrics import r2_score

print("Random Forest r2 Değeri: ",r2_score(Y,rf_reg.predict(X)))

# Ağaç kütüphanesi
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print("DecisionTree r2 Değeri: ", r2_score(Y,r_dt.predict(X)))
"""
Decision Tree R2 değeri 1 çıkmaktadır. Fakat bu net doğru karar vermemektedir
Random Forest R2 değeri 0.97 çıkmaktadır. Daha doğru bir karar vermektedir
"""

# Veri Ölçeklendirme - Scaler yapmak zorundayız
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

# SVR Model oluşturma
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf') #kernel parametresi önemli
svr_reg.fit(x_olcekli, y_olcekli)

print("SVR r2 değeri: ",r2_score(y_olcekli,svr_reg.predict(x_olcekli)))
"""SVR R2 değeri : 0.75 değerindedir. """














