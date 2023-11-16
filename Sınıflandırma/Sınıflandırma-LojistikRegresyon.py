# =============================================================================
# Lojistik Regresyon - Sınıflandırma 
# =============================================================================


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Verileri yükle
veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/veriler.csv")

# Bağımsız değişkenler (x) ve bağımlı değişken (y) seçimi
x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4].values

# Eğitim ve test setlerini oluştur
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Verileri ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Logistic Regression modelini oluştur
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = log_reg.predict(X_test)
print(y_pred)
print(" ")
print(y_test)
"""
y_pred
['e' 'e' 'e' 'e' 'k' 'e' 'e' 'e']
 y_test
['k' 'k' 'k' 'k' 'e' 'k' 'k' 'k']
"""

# Karmaşıklık Matrisi 
# Ölçümler metrics altında olur
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
"""
[[0 1]
 [7 0]]
"""












