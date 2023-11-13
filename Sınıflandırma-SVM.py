# SVM ve Çekirdek Hilesi - Kernel Trick


import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/veriler.csv")

x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

from sklearn.svm import SVC 
svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
print(y_pred)
print("")
print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
"""
y_pred
['e' 'e' 'e' 'e' 'k' 'e' 'e' 'k']
y_test
['k' 'k' 'k' 'k' 'e' 'k' 'k' 'k']

linear verildiğindeki sonuç aşağıdaki gibidir
[[0 1]
 [6 1]]

poly verildiğindeki sonuç aşağıdaki gibidir
[[1 0]
 [7 0]]

rbf verildiğindeki sonuç aşağıdaki gibidir
[[1 0]
 [5 2]]

sigmoid verildiğindeki sonuç aşağıdaki gibidir
[[0 1]
 [7 0]]

bu değerlere göre en iyi sonuca rbf ile elde edilmiştir
"""

