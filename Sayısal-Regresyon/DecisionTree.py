# KARAR AĞAÇLARI İLE TAHMİN 
# =============================================================================
# DECİSİON TREE
# iki boyutlu bir uzaya bölünür ve buna göre tahmin yapılır
# Örnek: Boydan veya kilodan bölünür ve sonrasında diğer değişkene bakılarak bölünür
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

# =============================================================================
# boy ve kilo'ya göre yas tahmini yapmak
# =============================================================================

# Ağaç kütüphanesi
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
Z=X+0.5
K=X-0.4
r_dt.fit(X,Y)

# =============================================================================
plt.scatter(X, Y)
plt.plot(X , r_dt.predict(X),color="red")
# =============================================================================
plt.plot(X , r_dt.predict(Z),color="green")
plt.plot(X , r_dt.predict(K),color="yellow")

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

# =============================================================================
# Net olarak bir maaş değeri gelir. 
# =============================================================================






































