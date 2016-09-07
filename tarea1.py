

# A)

# Descarga del dataset
import pandas as pd
import numpy as np
url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)

# Elimina la columna Unnamed 0, la cual es innecesaria porque ya se tiene una enumeracion.
df = df.drop('Unnamed: 0', axis=1)

# Extrae las filas de la columna train que tienen como valor "T" en forma de True y False,
# si es que corresponde a entrenamiento y test
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)

# Elimina la columna train, ya que anteriormente se extrajo la informacion y no se necesita.
df = df.drop('train', axis=1)


# B)

# Da informaciones varias. Shape() las dimensiones, info() descripcion de las cols y
# describe() da informaciones del dataset, tales como promedio, std, min ,etc.
df.shape
df.info()
df.describe()

# C)

# Se normalizan los datos para homogeneizar la varianza y obtener datos que no difieran
# mucho en valor entre las columnas
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']


# D)

import sklearn.linear_model as lm

# Se elimina la ultima columna (target)
X = df_scaled.ix[:,:-1]

# Es la dimension de cuantos registros (filas) hay en los datos
N = X.shape[0]

X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']
Xtrain = X[istrain]
ytrain = y[istrain]
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]

# El argumento fit_intercept dice si es necesario centrar los datos acorde al intercepto.
# Como nuestros datos ya estan normalizados, no es necesario, por lo que se setea False.
linreg = lm.LinearRegression(fit_intercept = False)

linreg.fit(Xtrain, ytrain)



# E)


# Coeficientes
coeficientes = linreg.coef_

# Z-score = Coeficientes/std
zdf = coeficientes/X.std()

# Que variables no existe suficiente evidencia para demostrar la relacion con la respuesta?
# No existe suficiente evidencia para todas las variables, excepto para el atributo lpsa?
# Esto porque seria la unica variable significante (|z-score| > 2)


# F)

# Comparar mes_test con mse_cv?

yhat_test = linreg.predict(Xtest)
mse_test = np.mean(np.power(yhat_test - ytest, 2))
from sklearn import cross_validation
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
k_fold = cross_validation.KFold(len(Xm),5)
print enumerate(k_fold)
mse_cv = 0

for k, (train, val) in enumerate(k_fold):
	linreg = lm.LinearRegression(fit_intercept = False)
	linreg.fit(Xm[train], ym[train])
	yhat_val = linreg.predict(Xm[val])
	mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
	mse_cv += mse_fold

mse_cv = mse_cv / 5


# E)

yhat_train = linreg.predict(Xtrain)
residuo = np.power(yhat_train - ytrain, 2)

# usar qqplot(residuo)











