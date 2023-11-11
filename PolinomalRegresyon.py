# =============================================================================
# Polinomal Regresyon - Polynominal Regresion
# Polinomlar karesel olarak gider.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Veri Yükleme
veriler = pd.read_csv('B:\GitHUB\BTK_Akademi_AI\Datasetler/maaslar.csv')

# Data Frame Dilimleme 

x = veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

# Tip Dönüşümü - Numpy Array'e dönüştürüldü
X = x.values
Y = y.values


# Linear(Doğrusal) Model oluşturma ve görselleştirme 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)
# =============================================================================
# plt.scatter(x,y,color='red')
# plt.plot(x,lr.predict(X),color='blue')
# =============================================================================



# Polynomial Regression (Dpğrusal olmayan) model oluşturma 
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2) #2. derece polinom

x_poly = pr.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(x_poly,y)


# 4. Dereceden polinom
pr = PolynomialFeatures(degree=4)
lr3 = LinearRegression()
x_poly2 = pr.fit_transform(X)
lr3.fit(x_poly2,y)

# =============================================================================
# Görselleştirme
# =============================================================================
plt.scatter(X, Y,color='red')
plt.plot(X,lr2.predict(pr.fit_transform(X)),color='blue')
plt.show()

plt.scatter(X, Y,color='red')
plt.plot(X,lr3.predict(pr.fit_transform(X)),color='blue')
plt.show()


# Tahminler
print("Eğitim Seviyesi 11: ",lr.predict([[11]]))
# Linear Reg için değeri= 34716
print("Eğitim Seviyesi 6.6: ",lr.predict([[6.6]]))
# Linear Reg için değeri = 16923

print("Eğitim Seviyesi 11: ",lr2.predict(pr.fit_transform([[11]])))
# Polynomial Reg için değeri = 89041
print("Eğitim Seviyesi 6.6: ",lr2.predict(pr.fit_transform([[6.6]])))
# Polynomial Reg için değeri = 8146 











































