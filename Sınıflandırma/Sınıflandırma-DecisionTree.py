# Karar Ağaçları ile Sınıflandırma

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler



# Verileri yükle
veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/veriler.csv")

# Bağımsız değişkenler (x) ve bağımlı değişken (y) seçimi
x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)

print(y_pred)
print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
"""
y_pred
['e' 'k' 'k' 'k' 'e' 'k' 'k' 'k']
y_test
['k' 'k' 'k' 'k' 'e' 'k' 'k' 'k']

criterion='entropy'
[[1 0]
 [1 6]]

criterion='gini'
[[1 0]
 [1 6]]
"""




















































