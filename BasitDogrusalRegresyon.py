# 3. Tahmin (Prediction) #1 - BASİT DOĞRUSAL REGRESYON

# =============================================================================
# 3.3 BASİT DIĞRUSAL REGRESYON
# Y = ax + b  fornülü vardır. 
# y--> bağımlı değişken 
# x --> bağımsız değişken  
# a --> kat sayı  
# AMAÇ= Grafik üzerindeki verilere en yakın geçen doğruyu bulmak 
# =============================================================================

# =============================================================================
# 3.4 VERİ YÜKLEME ve ÖN İŞLEME
# =============================================================================

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/satislar.csv")

aylar = veriler[['Aylar']]
satıslar = veriler[['Satislar']]

# Satislar Bağlı değişken, Aylar Bağımsız Değişken 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,
                                                 satıslar,
                                                 test_size=0.33,
                                                 random_state=0)
# Standartlaştırma işlemi yapılmadan da bulunabilir
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_Train = sc.fit_transform(x_train)
X_Test = sc.fit_transform(x_test) 
Y_Train = sc.fit_transform(y_train)
Y_Test = sc.fit_transform(y_test)
"""
# =============================================================================
# 3.5  - MODEL İNŞA ETMESİ
# =============================================================================

# Train ile eğitip Test kümesiyle test etmek amacımız

from sklearn.linear_model import LinearRegression
# Doğrusal Regresyon Modeli import edilmesi

lr = LinearRegression()

linear_model = lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

# Random State yapıldığından dolayı aylar sıralı değil.
# Bundan kaynaklı olarak da grafiik anlamsız olmaktadır
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
# x_test verilerini tahmin edilen değerleri gösterir
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")












