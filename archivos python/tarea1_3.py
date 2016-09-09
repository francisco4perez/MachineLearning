# Descarga del dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pylab as plt
from sklearn.linear_model import Lasso
url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)
df = df.drop('Unnamed: 0', axis=1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']

X = df_scaled.ix[:,:-1] #se le elimina la columna del target
N = X.shape[0]
X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa'] #columna target
Xtrain = X[istrain]
ytrain = y[istrain]
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]

Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
############################ CARGAR DATOS ##################################

# A)

# En la tercera linea se saca la columna "intercept", puesto que ya no se necesita.
# En la novena linea se ejecuta el metodo de Ridge a traves de factorizacion svd
# con el parametro fit_intercept igual a True, puesto que el mismo metodo se encarga del intercepto.

# Se puede observar del grafico que las caracteristicas que poseen mas peso son Lcavol, Lweight y svi
# Esto apoya las afirmaciones anteriores de los metodos FSS y BSS.

'''
#X = X.drop('intercept', axis=1)
Xtrain = X[istrain]
ytrain = y[istrain]
names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
alphas_ = np.logspace(4,-1,base=10)
coefs = []
model = Ridge(fit_intercept=True,solver='svd')
for a in alphas_:
	model.set_params(alpha=a)
	model.fit(Xtrain, ytrain)
	coefs.append(model.coef_)
ax = plt.gca()
for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
	print alphas_.shape
	print y_arr.shape
	plt.plot(alphas_, y_arr, label=label)
plt.legend()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Regularization Path RIDGE')
plt.axis('tight')
plt.legend(loc=2)
plt.show()




# B)

# Lasso al tener una penalizacion mas alta, es mas riguroso que Ridge al decir cual de las
# caracteristicas tiene mas peso. Nuevamente escogiendo a Lcavol, Lweight y svi

Xtrain = X[istrain]
ytrain = y[istrain]
names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
alphas_ = np.logspace(1,-2,base=10)
coefs = []
model = Lasso(fit_intercept=True)
for a in alphas_:
	model.set_params(alpha=a)
	model.fit(Xtrain, ytrain)
	coefs.append(model.coef_)
ax = plt.gca()
for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
	print alphas_.shape
	print y_arr.shape
	plt.plot(alphas_, y_arr, label=label)
plt.legend()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Regularization Path LASSO')
plt.axis('tight')
plt.legend(loc=2)
plt.show()


# C)

# Al analizar el grafico de los errores de test y entrenamiento en funcion de los alpha, se puede
# visualizar que el error de entrenamiento a medida que disminuye el valor de alpha, se observa que
# baja su valor, lo cual es esperado puesto que son los datos con los que se entreno el algoritmo.
# Tambien es esperado que el error de test empiece a ser mas alto que el de entrenamiento despues
# de cierto punto. Es importante notar que ambos errores empiezan a tener un valor constante
# despues de los valores de alpha 10e-1 y 10e-2.

Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
alphas_ = np.logspace(2,-2,base=10)
coefs = []
model = Ridge(fit_intercept=True)
mse_test = []
mse_train = []
for a in alphas_:
	model.set_params(alpha=a)
	model.fit(Xtrain, ytrain)
	yhat_train = model.predict(Xtrain)
	yhat_test = model.predict(Xtest)
	mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
	mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))
ax = plt.gca()
ax.plot(alphas_,mse_train,label='train error Ridge')
ax.plot(alphas_,mse_test,label='test error Ridge')
plt.legend(loc=2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
'''

# D)

Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
alphas_ = np.logspace(0.5,2,base=10)
coefs = []
model = Lasso(fit_intercept=True)
mse_test = []
mse_train = []
for a in alphas_:
	model.set_params(alpha=a)
	model.fit(Xtrain, ytrain)
	yhat_train = model.predict(Xtrain)
	yhat_test = model.predict(Xtest)
	mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
	mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))
ax = plt.gca()
ax.plot(alphas_,mse_train,label='train error Lasso')
ax.plot(alphas_,mse_test,label='test error Lasso')
plt.legend(loc=2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()

'''

# E)

def MSE(y,yhat): return np.mean(np.power(y-yhat,2))
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
k_fold = cross_validation.KFold(len(Xm),10)
best_cv_mse = float("inf")
model = Lasso(fit_intercept=True)
for a in alphas_:
model.set_params(alpha=a)
mse_list_k10 = [MSE(model.fit(Xm[train], ym[train]).predict(Xm[val]), ym[val]) \
for train, val in k_fold]
if np.mean(mse_list_k10) < best_cv_mse:
best_cv_mse = np.mean(mse_list_k10)
best_alpha = a
print "BEST PARAMETER=%f, MSE(CV)=%f"%(best_alpha,best_cv_mse)

'''








