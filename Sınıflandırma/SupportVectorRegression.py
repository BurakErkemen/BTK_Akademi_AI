# Support Vector Regression 
# =============================================================================
# Bir doğru ile ayrılacak ve o doğruya iki paralel doğru ile arasındaki
# maksimum değerleri alan en küçük marjine sahip doğruları bulmak amaçlanmıştır
# Marjinal (Aşırı aykırı) verilere karşı zaafiyeti vardır
# =============================================================================

import pandas as pd 
import matplotlib.pyplot as plt 

# Veri yükleme
veriler = pd.read_csv('B:\GitHUB\BTK_Akademi_AI\Datasetler/maaslar.csv')

# Data Frame Dilimleme 

x = veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

# Tip Dönüşümü - Numpy Array'e dönüştürüldü
X = x.values
Y = y.values

# Veri Ölçeklendirme - Scaler yapmak zorundayız
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

# SVR Model oluşturma
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf') #kernel parametresi önemli
svr_reg.fit(x_olcekli, y_olcekli)

# =============================================================================
# plt.scatter(x_olcekli, y_olcekli,color="red")
# plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")
# =============================================================================
# print(svr_reg.predict([[11]]))
# print(svr_reg.predict([[6.6]]))

# Tahmin Yapma 
tahmin_girdisi = [[11]]
tahmin_girdisi_olcekli = sc1.transform(tahmin_girdisi)
tahmin_olcekli = svr_reg.predict(tahmin_girdisi_olcekli)
tahmin_normal = sc2.inverse_transform(tahmin_olcekli.reshape(1, -1))

print("Ölçekli Tahmin:", tahmin_olcekli)
print("Normal Tahmin:", tahmin_normal)

from sklearn.metrics import r2_score
print("r2 Değeri: ",r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


# =============================================================================
# Çıktı
# rbf yerine poly değeriyle verilen çıktı daha beklenilesi bir değerdedir.
# Normal Tahmin: [[64028.60248]] 
# rbf değeri
# Normal Tahmin: [[27272.57392]]
# =============================================================================









































