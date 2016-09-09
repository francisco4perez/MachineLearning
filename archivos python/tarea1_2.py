import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt


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


from sklearn import cross_validation
# Funcion para hacer CrossValidation
def C_V(Xm,ym, K=10):
    k_fold = cross_validation.KFold(len(Xm),K)
    mse_cv = 0
    for k, (train, val) in enumerate(k_fold):
        linreg = lm.LinearRegression(fit_intercept = False)
        linreg.fit(Xm[train], ym[train])
        yhat_val = linreg.predict(Xm[val])
        mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
        mse_cv += mse_fold
    mse_cv = mse_cv / K
    return mse_cv

# A)

def fss(x, y, names_x, k = 10000):
    p = x.shape[1]-1
    k = min(p, k)
    names_x = np.array(names_x)
    remaining = range(0, p)
    selected = [p]
    current_score = 0.0
    best_new_score = 0.0
    while remaining and len(selected)<=k :
        score_candidates = []
        for candidate in remaining:
            model = lm.LinearRegression(fit_intercept=False)
            indexes = selected + [candidate]
            x_train = x[:,indexes]
            
            #calcular error mediante cross validation
            mse_candidate = C_V(x_train,y, K=10)
            score_candidates.append((mse_candidate, candidate))
        score_candidates.sort()
        score_candidates[:] = score_candidates[::-1]
        best_new_score, best_candidate = score_candidates.pop() #el de menor error es el mejor candidato
        remaining.remove(best_candidate)
        selected.append(best_candidate)
        print "selected = %s ..."%names_x[best_candidate]
        print "totalvars=%d, mse = %f"%(len(indexes),best_new_score)
    return selected
names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
sequence = fss(Xm,ym,names_regressors)
print sequence

# GRAFICAR error training y test en FSS
mse_trains = {}
mse_tests = {}
aux = []
for index in sequence:
    aux = aux + [index]
    model = lm.LinearRegression(fit_intercept=False)
          
    #calcular error de training set y test set
    x_train = Xtrain.as_matrix()[:,aux]
    predictions_train = model.fit(x_train, ym).predict(x_train)
    residuals_train = predictions_train - ym
    mse_train = np.mean(np.power(residuals_train, 2))
    mse_trains[len(aux) -1] = mse_train
        
    x_test = Xtest.as_matrix()[:,aux]
    y_test = ytest.as_matrix()
    predictions_test = model.fit(x_train, ym).predict(x_test)
    residuals_test= predictions_test - y_test
    mse_test = np.mean(np.power(residuals_test, 2))
    mse_tests[len(aux)-1] = mse_test

mse_trains.pop(0) #sin contar 0 variables
mse_tests.pop(0) #sin contar 0 variables
ax = plt.gca()
ax.plot(range(1,9),mse_trains.values(), label='train error')
ax.plot(range(1,9),mse_tests.values() , label='test error')
plt.legend(loc=2)
plt.xlabel('number variable')
plt.ylabel('mean square error')
plt.title('Error on FSS')
plt.axis('tight')
plt.show()

# B)

def bss(x, y, names_x, k = 10000):
    p = x.shape[1]-1 #numero de caracteristicas
    k = min(p, k)
    names_x = np.array(names_x)
    removing = [] #orden en que se eliminan
    selected =  range(0, p) #cambio
    current_score = 0.0
    best_new_score = 0.0
    while len(selected)>0 : #cambio
        score_candidates = [] #candidatos a ser eliminados
        for candidate in selected:
            model = lm.LinearRegression(fit_intercept=False)
            
            indexes = [p] + selected  # p intercepto
            indexes.remove(candidate) #elimina el posible candidato a ser eliminado indexes = selected - candidate
            x_train = x[:,indexes] #datos a probar
            
            #calcular error mediante cross validation
            mse_candidate = C_V(x_train,y, K=10)
            score_candidates.append((mse_candidate, candidate))
            #print "viendo que tal es eliminar: "+ str(candidate)+ " entrega un error de: "+str(mse_candidate)
        score_candidates.sort()
        score_candidates[:] = score_candidates[::-1]
        #se elige el modelo con el menor error localmente, elimmando el candidato que fue elimiando de ese modelo
        best_new_score, best_candidate = score_candidates.pop() 
        selected.remove(best_candidate)  #cambio
        removing.append(best_candidate) #cambio
        print "selected to delete = %s ..."%names_x[best_candidate]
        print "totalvars=%d, mse = %f"%(len(indexes),best_new_score)
    return removing
names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
sequence_bss = bss(Xm,ym,names_regressors) #secuencia de eliminacion
print sequence_bss

#GRAFICAR error training set y test set en BSS
aux = [Xm.shape[1]-1] + sequence_bss
for index in sequence_bss:
    model = lm.LinearRegression(fit_intercept=False)

    #calcular error de training set y test set
    x_train = Xtrain.as_matrix()[:,aux]
    predictions_train = model.fit(x_train, ym).predict(x_train)
    residuals_train = predictions_train - ym
    mse_train = np.mean(np.power(residuals_train, 2))
    mse_trains[len(aux) -1] = mse_train
        
    x_test = Xtest.as_matrix()[:,aux]
    y_test = ytest.as_matrix()
    predictions_test = model.fit(x_train, ym).predict(x_test)
    residuals_test= predictions_test - y_test
    mse_test = np.mean(np.power(residuals_test, 2))
    mse_tests[len(aux)-1] = mse_test
    
    aux.remove(index)
ax = plt.gca()
ax.plot(range(1,9),mse_trains.values(), label='train error')
ax.plot(range(1,9),mse_tests.values() , label='test error')
plt.legend(loc=2)
plt.xlabel('number variable')
plt.ylabel('mean square error')
plt.title('Error on BSS')
plt.axis('tight')
plt.show()