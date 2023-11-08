# 4. Tahmin - COKLU DOGRUSAL REGRESYON 

# =============================================================================
# **Birden fazla değişken (parametre) kullanılacaksa 
# Basit Doğrusal Regresyon kullanılamaz.
# **Çok değişken olması durumunda hangisi işimize yarıyor acaba sorusunun sorulması 
# halinde bazı çözüm yolları çıkmaktadır. 
# --> Kukla Değişken(Dummy Variable) 
# -->P-Value (Olasılık Değeri)
# -->Geri Eleme(Backward Elimination)
# =============================================================================


# =============================================================================
# Basit Doğrusal Reg. formülü => yi = axi + b + ei iken 
# Çoklu Doğrusal Reg. formülü => y= B0 + B1X1 + ... + BnXn + e 
# örnek olarak: Boy = a + b(kilo) + c(yaş) + d(ayakkabı no) + e 
# =============================================================================


# =============================================================================
# Dummy Variable - Kukla Değişken 
# Dummy ile dönüştürülen değerleri aynı anda olmaması gerekli 
# Örnek olarak cinsiyet, kadın ve erkek olduğunda cinsiyet ve alınan kolon alınır
# kadın erkek kolonları 0 1 olarak yazılır. Sadece bir değeri almak mantıklıdır
# =============================================================================


# =============================================================================
# P-VALUE - OLASILIK DEĞERİ
# Hipotez üzerinden gidilir. 
# H0 = Sıfır Hipotezi (Başlangıc Hipotezi)
# H1 = Alternatif Hipotez. H0'ın karşıt hipotezidir
# p-değeri genellikle 0.05 alınır. 
# p değeri azaldıkça H0 hatalı olma ihtimali +
# p-değeri büyüdükçe H1 hatalı olma ihtimali artar
# =============================================================================

# =============================================================================
# Çoklu Doğrusal Regresyonda Değişken Seçimi Nasıl Yapılır?
# Değişkenlerin etkileme miktarını aynı oranda mı etkilemekte?
# ****Bütün Değişkenleri Dahil Etmek
# Bütün değişkenler sistemde kullanılır. Başlangıç olarak fikir vermeye sağlar
# ****Geriye Doğru Eleme (Backward Elimination)
# Bütün değerler eklenir ve başta belirlenen değer sağlanana kadar sütunlar elenir
# P-Value değerine göre sistem sürekli yenilenerek ilerler. Başlangıç değerinin 
# altına inene kadar ayarlanır
# ****İleriye Seçim (Forward Selection)
# Yukarıdakinden farkı tek bir değişken vardır.
# Başta belirlenen kriter sağlanana kadar sütunlar eklenir.
# ****Çift Yönlü Eleme(Bidirectional Elemination)
# Başlangıçta değer belirlenir-Significance Level(SL) Yukarıda da kullanıldı.
# En düşük p-value değerine sahip değişken ele alınır. 
# Bütün değişkenler sisteme eklenir. 
# ****Skor Karşılaştırma (Score Comprasion)
# SL değerini kişisel olarak belinir. Başarı kriteri belirlenir
# Bütün olası regresyon modelleri inşa edilir. Belirtilen kriteri en iyi sağlayan
# yöntem seçilir
# =============================================================================

# VERİLERİN HAZIRLANMASI
# cinsiyet ve ülke encode yapılmalı 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/veriler.csv")
ulke = veriler.iloc[:,:1].values
cinsiyet = veriler.iloc[:,-1:].values
yas = veriler.iloc[:,1:4].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ulke = ohe.fit_transform(ulke).toarray()

cinsiyet[:,-1] = le.fit_transform(veriler.iloc[:,-1])
cinsiyet = ohe.fit_transform(cinsiyet).toarray()


sonuc = pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
sonuc2 = pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
sonuc3 = pd.DataFrame(data=cinsiyet[:,:-1],index=range(22),columns=['cinsiyet'])

s = pd.concat([sonuc,sonuc2],axis=1)
s2 = pd.concat([s,sonuc3],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s, sonuc3,random_state=0,train_size=0.33)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)


boy = s2.iloc[:,3:4].values
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)
# print(y_pred,"\n",y_test)


# =============================================================================
# PYTHON İLE GERİ ELEME
# =============================================================================
import statsmodels.api as sm
X = np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values 
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())


X_l = veri.iloc[:,[0,1,2,3,5]].values 
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())



X_l = veri.iloc[:,[0,1,2,3]].values 
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())




















