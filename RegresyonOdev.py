# Odev2 
"""
***Yapılacaklar Listesi
*Gerekli/Gereksiz Bağımsız Değişkenler
*5 Farklı Yönteme Göre Regresyon Çıkartmak - MLR,-PR,-SVR,-DT,-RR
*Yöntem Başarılarını Karşılaştırma
*10 yıl tecrübeli ve 100 puan almış bir Ceo ve aynı özelliklere sahip
Müdürün maaşlarının 5 yönteme göre tahmin sonuçlları
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor





veriler = pd.read_csv('B:\GitHUB\BTK_Akademi_AI\Datasetler/maaslar_yeni.csv')


# Dogrusal Regresyon
# Veri Önişleme
x = veriler.iloc[:,2:5] # Bağımsız değişken
y = veriler.iloc[:,5:] # Bağımlı Değğişken
X = x.values
Y = y.values 

# numerical_columns = veriler.select_dtypes(include=['float64', 'int64'])
# correlation_matrix = numerical_columns.corr()
print( veriler.select_dtypes(include=['float64', 'int64']).corr())

# Linear Regresyon Model kurulumu
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# OLS Değerlerinin Hesaplanması 
model1 = sm.OLS(lin_reg.predict(X),X)
print("\n","Model1")
print(model1.fit().summary())
"""
TEK PARAMETRELİ
x = veriler.iloc[:,2:3] # Bağımsız değişken
R-squared (uncentered):                   0.942
Adj. R-squared (uncentered):              0.940

ÇOK PARAMETRELİ
x = veriler.iloc[:,2:5] 
R-squared (uncentered):                   0.903
Adj. R-squared (uncentered):              0.892
"""
# print(r2_score(Y,lin_reg.predict(X))) #0.52


# Çoklu Doğrusal Regresyon

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print("\n","Model2")
print(model2.fit().summary())
"""
TEK PARAMETRELİ
degree = 2 
R-squared (uncentered):                   0.810
R-squared (uncentered):                   0.803

degree = 4
R-squared (uncentered):                   0.759
Adj. R-squared (uncentered):              0.751

ÇOK PARAMETRELİ
degree = 2 
R-squared (uncentered):                   0.729
R-squared (uncentered):                   0.698

degree = 4
R-squared (uncentered):                   0.680
Adj. R-squared (uncentered):              0.644
"""


# SVR 
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

svr_reg = SVR(kernel='rbf') #kernel parametresi önemli
svr_reg.fit(x_olcekli, y_olcekli)

model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print("\n","Model3")
print(model3.fit().summary())

"""
TEKPARAMETRELİ
R-squared (uncentered):                   0.770
Adj. R-squared (uncentered):              0.762

ÇOK PARAMETRELİ
R-squared (uncentered):                   0.782
Adj. R-squared (uncentered):              0.758
"""

# DecisionTree 
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

model4 = sm.OLS(r_dt.predict(X),X)
print("\n","Model4")
print(model4.fit().summary())

"""
TEK PARAMETRELİ
R-squared (uncentered):                   0.751
Adj. R-squared (uncentered):              0.742

ÇOK PARAMETRELİ
R-squared (uncentered):                   0.679
Adj. R-squared (uncentered):              0.644
"""


# RANDOM FOREST
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

model5 = sm.OLS(rf_reg.predict(X),X)
print("\n","Model5")
print(model5.fit().summary())

"""
TEK PARAMETRELİ
R-squared (uncentered):                   0.719
Adj. R-squared (uncentered):              0.709

ÇOK PARAMETRELİ
R-squared (uncentered):                   0.713
Adj. R-squared (uncentered):              0.681

"""






