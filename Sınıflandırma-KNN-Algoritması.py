# K-NN (K-Nearest Neightborhood) AlgoritmasÄ±
# =============================================================================
# Ä°ki deÄŸer arasÄ±ndaki Ã¶lÃ§Ã¼mler hesabÄ±yla yapÄ±lÄ±r 
# 
# =============================================================================

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/veriler.csv")

# Baðýmsýz deðiþkenler (x) ve baðýmlý deðiþken (y) seçimi
x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train, y_train)

y_pred= knn.predict(X_test)
print(y_pred)
print(" ")
print(y_test)
# Karmaþýklýk Matrisi 
# Ölçümler metrics altýnda olur
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
"""
y_pred
['k' 'k' 'e' 'e' 'e' 'e' 'e' 'k']
 y_test
['k' 'k' 'k' 'k' 'e' 'k' 'k' 'k']

n_neighbors = 3 iken
[[1 0]
 [4 3]]
1 tane doðru deðeri doðru vermiþtir
3 tane yanlýþ deðeri yanlýþ vermiþtir fakat 4 tane yanlýþ deðeri doðru vermýþtir.

n_neighbors = 1 iken
[[1 0]
 [1 6]]
1 tane doðru deðeri doðru vermiþtir
6 tane yanlýþ deðeri yanlýþ vermiþtir fakat 1 yane yanlýþ deðeri doðru vermiþtir

Böylece, n_neighbors deðeri 3ten1 e çekildiðinde öðrenme artmýþtýr
ve yanlýþlýk azalmýþtýr
"""




























