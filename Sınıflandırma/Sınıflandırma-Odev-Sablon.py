# Sınıflandırma Algoritmaları - Ödev


import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Modellerin Gelişimi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
veriler = pd.read_excel("B:\GitHUB\BTK_Akademi_AI\Datasetler/Iris.xls")

x = veriler.iloc[:, :4].values
y = veriler.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


# Logistic Regression modelini oluştur
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = log_reg.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("LOGİSTİC")
print(cm)


# KNN 
knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("KNN")
print(cm)



# SVM
svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("SVM")
print(cm)


# Decision Tree
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("Decision Tree")
print(cm)



# Random Forest
rfc = RandomForestClassifier(n_estimators=5,criterion='entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("RANDOM FOREST")
print(cm)



# Naif Bayes 
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("Naif Bayes")
print(cm)


# Farklı Verilerle Test Etme

X_test2 = scaler.transform([[2,3,2.5,4]])
y_pred2 = gnb.predict(X_test2)
print("Naif Bayes Predict 2,3,2.5,4")
print(y_pred2)

X_test3 = scaler.transform([[2,3,2.5,4]])
y_pred3 = dtc.predict(X_test2)
print("Decision Tree Predict 2,3,2.5,4")
print(y_pred3)

# ROC, TPR, FPR DEĞERLERİ
# =============================================================================
# from sklearn import metrics
# y_proba = log_reg.predict_proba(X_test)
# fpr,tpr,thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
# print(fpr)
# print(tpr)
# =============================================================================




k e e e k k k k 
e k k k e e e e 





















