# K-NN (K-Nearest Neightborhood) Algoritması
# =============================================================================
# İki değer arasındaki ölçümler hesabıyla yapılır 
# 
# =============================================================================

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/veriler.csv")

# Ba��ms�z de�i�kenler (x) ve ba��ml� de�i�ken (y) se�imi
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
# Karma��kl�k Matrisi 
# �l��mler metrics alt�nda olur
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
1 tane do�ru de�eri do�ru vermi�tir
3 tane yanl�� de�eri yanl�� vermi�tir fakat 4 tane yanl�� de�eri do�ru verm��tir.

n_neighbors = 1 iken
[[1 0]
 [1 6]]
1 tane do�ru de�eri do�ru vermi�tir
6 tane yanl�� de�eri yanl�� vermi�tir fakat 1 yane yanl�� de�eri do�ru vermi�tir

B�ylece, n_neighbors de�eri 3ten1 e �ekildi�inde ��renme artm��t�r
ve yanl��l�k azalm��t�r
"""




























