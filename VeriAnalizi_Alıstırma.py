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
# 2.6 - KATEGORİK VERİLER
# **Kategorik veri örneği: Kadın mı? Erkek mi?
# **Kategorik Veri --> 1. Nominal 2. Ordinal
# *1.Nominal = Sıralama Yapılamaz ve ilişki kurulamaz. Cep telefonu markası gibi
# *2.Ordinal = Sıralama yapılabilir örneğin şehir plaka kodları
# **Sayısal Veri --> 1. Oransal 2. Aralık
# *1.Oransal = birbirine çapılıp bölünebilen dört işleme müsait
# *2.Aralık = Dört işleme müsait olmayan bir aralığın içindeymiş gibi olan verilerdir
# **Kategorik veriler sayıya çevirilerek eğitilme işleminde kullanılırlar ve model 
# eğitiminde rol oynarlar. 
# =============================================================================
ulke = veriler.iloc[:,0:1].values # Ulke column gelir
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
#  sütunun değerlerini sayısal bir forma dönüştürmek için LabelEncoder kullanır
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
# print(ulke)

ohe = preprocessing.OneHotEncoder()
# Ardından, OneHotEncoder kullanılır. OneHotEncoder, kategorik verilerin sayısal 
# dönüşümünü alır ve her kategori için ayrı bir sütun oluşturur. Her satır, ilgili 
# kategoriye aitse, o sütun için 1 değerini alırken, diğer sütunlar için 0 değerini alır.
ulke = ohe.fit_transform(ulke).toarray()
# print(ulke)

# =============================================================================
# 2.7 - VERİLERİN BİRLEŞTİRİLMESİ ve DATAFRAME OLUSUMU
# =============================================================================
sonuc_ulke_df = pd.DataFrame(data=ulke,index = range(22), columns=['fr','tr','us'])
# print(sonuc_ulke_df)
sonuc_yas_df = pd.DataFrame(data=yas,index = range(22), columns=['boy','kilo','yas'])
# print(sonuc_yas_df)

cinsiyet = veriler.iloc[:,-1].values
sonuc_cinsiyet_df=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
# print(sonuc_cinsiyet_df)

# DATAFRAME BİRLEŞTİRME 
s = pd.concat([sonuc_yas_df,sonuc_ulke_df],axis=1)
# print(s)

s_son = pd.concat([s,sonuc_cinsiyet_df],axis=1)
print(s_son) # Bu çıktı ile datasetimiz güncellendi 

# =============================================================================
# 2.8 - VERİ KÜMESİNİN EĞİTİM ve TEST İÇİN BÖLÜNMESİ
# =============================================================================

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s
                                                 ,sonuc_cinsiyet_df
                                                 ,test_size=0.33
                                                 ,random_state=0)
# x değeri bağımsız değişkenler --> cinsiyet dışındakiler
# y değeri bağıumlı değişkenler --> cinsiyet

# =============================================================================
# 2.9 - ÖZNİTELİK ÖLÇEKLEME
# =============================================================================
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_Train = sc.fit_transform(x_train)
X_Test = sc.fit_transform(x_test)


# =============================================================================
# 2.10 - VERİ ÖN İŞLEME ŞABLONU
# =============================================================================










