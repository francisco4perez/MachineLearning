
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

yhat_test = linreg.predict(Xtest)
mse_test = np.mean(np.power(yhat_test - ytest, 2))
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()

print "error test: %f"%(mse_test)
print "error cross-validation K = 10: %f"%(C_V(Xm,ym))
print "error cross-validation K=5: %f"%(C_V(Xm,ym,5))
ep = (C_V(Xm,ym) - mse_test)/C_V(Xm,ym)
print "error porcentual de K =10 y K = 5"
print ep*100
ep = (C_V(Xm,ym,5) - mse_test)/C_V(Xm,ym,5)
print ep*100




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
		predictions_train = model.fit(x_train, y).predict(x_train)
		residuals_train = predictions_train - y
		mse_candidate = np.mean(np.power(residuals_train, 2))
		score_candidates.append((mse_candidate, candidate))
	score_candidates.sort()
	score_candidates[:] = score_candidates[::-1]
	best_new_score, best_candidate = score_candidates.pop()
	remaining.remove(best_candidate)

	selected.append(best_candidate)
	print "selected = %s ..."%names_x[best_candidate]
	print "totalvars=%d, mse = %f"%(len(indexes),best_new_score)
return selected


names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
fss(Xm,ym,names_regressors)



