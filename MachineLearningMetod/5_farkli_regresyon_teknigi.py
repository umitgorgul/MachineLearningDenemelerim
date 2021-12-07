import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
#r^2 degerleri için
from sklearn.metrics import r2_score

#veri yükleme
veriler = pd.read_csv('maaslar_yeni.csv')

#data frame dilimleme (slice)
x = veriler.iloc[:, 2:5]
y = veriler.iloc[:, 5:]

print(veriler.corr())

#numpy array donusumu
X = x.values
Y = y.values


#lineer regresyon dogrusal model olusturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

model = sm.OLS(lin_reg.predict(X), X)
print(model.fit().summary())



#polynomal regresyon non linear model olusturma
from sklearn.preprocessing import PolynomialFeatures

#4.derecen polinom
polyReg4 = PolynomialFeatures(degree = 4)
x_poly4 = polyReg4.fit_transform(X)
linReg4 = LinearRegression()
linReg4.fit(x_poly4, y)


print('poly ols')
model2 = sm.OLS(linReg4.predict(polyReg4.fit_transform(X)), X)
print(model2.fit().summary())


print('polynomial R^2 degeri 4')
print(r2_score(Y, linReg4.predict(polyReg4.fit_transform(X)))) # y_olcekli=TRUE svr_reg.predict(x_olcekli)s=PREDICT


#verilerin ölceklendirilmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)


print('svr ols')
model3 = sm.OLS(svr_reg.predict(x_olcekli), x_olcekli)
print(model3.fit().summary())

print('svr R^2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))# y_olcekli=TRUE svr_reg.predict(x_olcekli)s=PREDICT

#DecisionTree
from sklearn.tree import DecisionTreeRegressor
dt_r = DecisionTreeRegressor(random_state=0)
dt_r.fit(X,Y)



print('dt ols')
model4 = sm.OLS(dt_r.predict(X), X)
print(model4.fit().summary())

print('Decision tree R^2 degeri')
print(r2_score(Y, dt_r.predict(X)))# Y=TRUE X=PREDICT

#RandomForest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y.ravel())



print('rf ols')
model5 = sm.OLS(rf_reg.predict(X), X)
print(model5.fit().summary())

print('RANDOM FOREST R^2 degeri')
print(r2_score(Y, rf_reg.predict(X)))#Y = true


#ozet r2 degereleri
print('polynomial R^2 degeri 4')
print(r2_score(Y, linReg4.predict(polyReg4.fit_transform(X)))) # y_olcekli=TRUE svr_reg.predict(x_olcekli)s=PREDICT

print('svr R^2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))# y_olcekli=TRUE svr_reg.predict(x_olcekli)s=PREDICT

print('Decision tree R^2 degeri')
print(r2_score(Y, dt_r.predict(X)))# Y=TRUE X=PREDICT

print('RANDOM FOREST R^2 degeri')
print(r2_score(Y, rf_reg.predict(X)))#Y = true