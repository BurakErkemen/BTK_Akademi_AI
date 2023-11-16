# =============================================================================
# Polinomal Regresyon - Polynominal Regresion
# Polinomlar karesel olarak gider.
# =============================================================================

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri Yükleme
veriler = pd.read_csv('B:\GitHUB\BTK_Akademi_AI\Datasetler/maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# Görselleştirme

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

# Tahminler
print("Eğitim Seviyesi 11: ",lin_reg.predict([[11]]))
# Linear Reg için değeri= 34716
print("Eğitim Seviyesi 6.6: ",lin_reg.predict([[6.6]]))
# Linear Reg için değeri = 16923

print("Eğitim Seviyesi 11: ",lin_reg2.predict(poly_reg.fit_transform([[11]])))
# Polynomial Reg için değeri = 89041
print("Eğitim Seviyesi 6.6: ",lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
# Polynomial Reg için değeri = 8146 

from sklearn.metrics import r2_score
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))
# R2 değeri 0.99















