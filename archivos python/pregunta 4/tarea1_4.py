import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
def MSE(y,yhat): return np.mean(np.power(y-yhat,2))

def best_alpha(alphas_, X,y, model, X_val,y_val):
    best_cv_mse = float("inf")
    for a in alphas_:
        model.set_params(alpha=a)
        mse =  MSE(model.fit(X,y).predict(X_val) , y_val)
        if mse < best_cv_mse:
            best_cv_mse = mse
            best_alpha = a
    return best_alpha

print "hola.. estamos realizando la operacion"

#para los datos con columna oscar
X_test = csr_matrix(mmread('ngrams-deprels-fp1-origin.runtime.budget.numscreen.ratings.seasons.stars/test.x.mm'))
y_test = np.loadtxt('ngrams-deprels-fp1-origin.runtime.budget.numscreen.ratings.seasons.stars/test.y.dat')

X_train = csr_matrix(mmread('ngrams-deprels-fp1-origin.runtime.budget.numscreen.ratings.seasons.stars/train.x.mm'))
y_train = np.loadtxt('ngrams-deprels-fp1-origin.runtime.budget.numscreen.ratings.seasons.stars/train.y.dat')

X_val = csr_matrix(mmread('ngrams-deprels-fp1-origin.runtime.budget.numscreen.ratings.seasons.stars/dev.x.mm'))
y_val = np.loadtxt('ngrams-deprels-fp1-origin.runtime.budget.numscreen.ratings.seasons.stars/dev.y.dat')

print "termino de cargar la data con menos caracteristicas (ya filtrada)"

import sklearn.linear_model as lm

model = lm.LinearRegression(fit_intercept = False)
model.fit(X_train,y_train)
best_score = model.score(X_test,y_test)
print "Score de regresion ordinaria: %f"%(best_score)

model = lm.Ridge(fit_intercept = False )
alphas_ = np.logspace(2,5, base = 10)
a = best_alpha(alphas_,X_train,y_train,model,X_val,y_val)
model.set_params(alpha=a)
model.fit(X_train,y_train)
score = model.score(X_test, y_test)
print "Score de regresion Ridge: %f , con alpha: %f"%(score,a)

model = lm.Lasso(fit_intercept = False ,tol = 0.0000001)
alphas_ = np.logspace(7,4, base= 10)
a = best_alpha(alphas_, X_train,y_train, model, X_val,y_val)  #a = 255954.792270 
model.set_params(alpha=a)
model.fit(X_train,y_train)
score = model.score(X_test, y_test)
print "Score de regresion Lasso: %f , con alpha: %f"%(score,a)

## analisis de error para Ridge
alphas_ = np.logspace(7,-2,base=10)
coefs = []
model = lm.Ridge(fit_intercept=False)
mse_test = []
mse_train = []
mse_val = []
for a in alphas_:
    model.set_params(alpha=a)
    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)
    mse_train.append(MSE(yhat_train,y_train))
    mse_test.append(MSE(yhat_test, y_test))
    mse_val.append(MSE(yhat_val,y_val))
import matplotlib.pylab as plt
ax = plt.gca()
ax.plot(alphas_,mse_train,label='train error Ridge')
ax.plot(alphas_,mse_val,label= 'cross-validation error Ridge')
ax.plot(alphas_,mse_test,label='test error Ridge')
plt.legend(loc=2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()

## analisis de error para Lasso
alphas_ = np.logspace(8,-2,base=10)  # 8 a 0
coefs = []
model = lm.Lasso(fit_intercept=False, tol =0.0000001)
mse_test = []
mse_train = []
mse_val = []
for a in alphas_:
    model.set_params(alpha=a)
    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)
    mse_train.append(MSE(yhat_train,y_train))
    mse_test.append(MSE(yhat_test, y_test))
    mse_val.append(MSE(yhat_val,y_val))
import matplotlib.pylab as plt
ax = plt.gca()
ax.plot(alphas_,mse_train,label='train error Lasso')
ax.plot(alphas_,mse_val,label= 'cross-validation error Lasso')
ax.plot(alphas_,mse_test,label='test error Lasso')
plt.legend(loc=2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()


###################################################################################################3
## dataset con mas caracteristicas
X_test = csr_matrix(mmread('ngrams-deprels-origin.runtime.budget.numscreen.ratings.seasons/test.x.mm'))
y_test = np.loadtxt('ngrams-deprels-origin.runtime.budget.numscreen.ratings.seasons/test.y.dat')

X_train = csr_matrix(mmread('ngrams-deprels-origin.runtime.budget.numscreen.ratings.seasons/train.x.mm'))
y_train = np.loadtxt('ngrams-deprels-origin.runtime.budget.numscreen.ratings.seasons/train.y.dat')

X_val = csr_matrix(mmread('ngrams-deprels-origin.runtime.budget.numscreen.ratings.seasons/dev.x.mm'))
y_val = np.loadtxt('ngrams-deprels-origin.runtime.budget.numscreen.ratings.seasons/dev.y.dat')

print "termino de cargar la data"

import sklearn.linear_model as lm

model = lm.LinearRegression(fit_intercept = False)
model.fit(X_train,y_train)
best_score = model.score(X_test,y_test)
print "Score de regresion ordinaria: %f"%(best_score)

model = lm.Ridge(fit_intercept = False )
alphas_ = np.logspace(4,9, base = 10)
a = best_alpha(alphas_,X_train,y_train,model,X_val,y_val)
model.set_params(alpha=a)
model.fit(X_train,y_train)
score = model.score(X_test, y_test)
print "Score de regresion Ridge: %f , con alpha: %f"%(score,a)

model = lm.Lasso(fit_intercept = False ,tol = 0.0000001)
alphas_ = np.logspace(7,4, base= 10)
a = best_alpha(alphas_, X_train,y_train, model, X_val,y_val)  #a = 255954.792270 
model.set_params(alpha=a)
model.fit(X_train,y_train)
score = model.score(X_test, y_test)
print "Score de regresion Lasso: %f , con alpha: %f"%(score,a)