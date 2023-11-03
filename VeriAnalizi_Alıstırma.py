# =============================================================================
# # 2.3 - VERİ YÜKLENİMİ
# =============================================================================
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

veriler = pd.read_csv('B:\GitHUB\BTK_Akademi_AI/veriler.csv')
veriler_boy = veriler[['boy']]
# print(veriler,"\n")
# print(veriler_boy)

# =============================================================================
# 2.4 Python ve Nesne Yönelimli Programlama
# =============================================================================
# classlar
class insan:
    boy = 170
    def kosmak(self,b):
        return b**2 
ali = insan()
print(ali.boy , ali.kosmak(5))

# listeler
liste = [1,2,3]
print(liste[1]) 

# =============================================================================
# 2.5 - EKSİK VERİLER   
# =============================================================================
eksik_veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI/eksikveriler.csv")
from sklearn.impute import SimpleImputer 
# Nan değerleri bir şekilde doldurmak için kullanılır
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
# strategy=' ' fonksiyonu impute edilen şeylere ne yapılacağı stratejisini belirlemektir.
yas = eksik_veriler.iloc[:,1:4].values
# iloc[row baslangıc ve bitis degeri,sütun baslangıc ve bitis değeri]
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
# **imputer fit fonksiyonu ile dizi içindeki istenilen değerler öğretiliyor
# daha sonrasında transform fonk ile bu değerler dizinin istenilen row'una atılıyor

# =============================================================================
# KATEGORİK VERİLER
# =============================================================================

