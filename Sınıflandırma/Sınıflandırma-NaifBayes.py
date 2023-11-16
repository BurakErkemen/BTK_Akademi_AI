# Bayes Teoremi - Naive Bayes S覺n覺fland覺rma Algoritmas覺 
# =============================================================================
# 1. Gaussian Naive Bayes
# Gaus dağılımı üzerinden çalışır
# Dataset = sürekli bir değerse, reel-ondalıklı sayı olabiliyorsa 
# 2. Multinominal Naive Bayes
# Multinominal dağılımı üzerinden çalışır
# Dataset = Birbirinden farklı değerler olanlarda, Araba markaları gibi
# 3. Bernouilli Naive Bayes
# Bernouilli dağılımı üzerinden çalışır
# Dataset = Binary dağıtımı kullanır. 0 1 gibi değerlerde kullanılır. 
# sigara içiyor-içmiyor değerleri gibi
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler



# Verileri yükle
veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/veriler.csv")

# Bağımsız değişkenler (x) ve bağımlı değişken (y) seçimi
x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4].values


# Eğitim ve test setlerini oluştur
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
"""
8 değerden sadece 1 doğru sınıflandırma bulunmaktadır
[[0 1]
 [6 1]]
"""
print(y_test)
print(y_pred)

"""
y_test
['k' 'k' 'k' 'k' 'e' 'k' 'k' 'k']
y_pred
['e' 'e' 'e' 'e' 'k' 'e' 'e' 'k']
"""






















