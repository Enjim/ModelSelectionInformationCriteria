import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
# statsmodels.tsa.arima_model.ARMA
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.optimize import minimize, fmin_slsqp
from statsmodels.tools.numdiff import approx_hess

import pandas as pd 
import numpy as np
import statsmodels.tsa.api as smt
n=15
# Define the parameters of the model
omega = 0.0734
alpha = 0.0776
beta = 0.8866

#AR Params & MA Params
ar_params = [0, 0.5167, 0.0142,-0.5778]
ma_params = [ -0.443, -0.0882, 0.7276, -0.1108]
p = len(ar_params)
q = len(ma_params)

# Generate a white noise error term
np.random.seed(100)
errors = np.random.normal(0, 1, n)

# Initialize the conditional variances
h = np.zeros(n)
y = np.zeros(n)
mu = np.zeros(n)
h[0] = np.var(errors)

if p>q:
    condition = p 
else: 
    condition = q 
condition = condition + 1 
    
y[:condition] = np.array([0.1, -0.1, 0.1, -0.1, 0.1])
h[:condition] = np.ones(condition)- 0.8

# Simulate the GARCH(1,1) process
for i in range(condition, n):
    h[i] = omega + alpha * errors[i-1]**2 + beta * h[i-1]
    errors[i] = np.sqrt(h[i]) * errors[i]
    AR = ar_params[0]
    MA = 0
    for p in range(1,len(ar_params)):
        AR += ar_params[p] * y[i-p]
    for q in range(len(ma_params)):
        MA += ma_params[q] * errors[i-1-q]
    y[i] = AR + MA + errors[i]
    mu[i] = AR + MA
    AR = 0 

plt.plot(y)
plt.show()

def calcArma(params, rets):
    np.random.seed(1)
    #epsilon = np.random.normal(0,1,n)
    ar_param = params[:P]
    ma_param = params[P:P+Q]
    garch_param = params[P+Q:]
    # print('AR PARAM : ', ar_param)
    # print("MA PARAM : ", ma_param)
    # print("GARCH PARAM : ", garch_param)
    #print(len(ar_param), len(ma_param), len(garch_param))
    alpha = garch_param[0]
    beta = garch_param[1]
    omega = garch_param[2]
    
    p = len(ar_param)
    q = len(ma_param)
    #print(q)
    mu = np.zeros(n)
    #y = np.zeros(n)
    resid = np.zeros(n)

    MA = ma_param[0]
    MU = ar_param[0]
    
    if p>q:
        condition = p 
    else: 
        condition = q 
        
    np.random.seed(50)
    mu[:condition] = np.random.rand(condition)
    resid[:condition] = np.random.rand(condition)
    sigma = np.zeros(n)
    sigma[:condition] = abs(np.random.rand(condition))

    for i in range(condition,n):
        MU += ar_param[0]
        for j in range(q):
            MA += ma_param[j] * resid[i-q+j]
        for r in range(1,p):
            MU += ar_param[r]*rets[i-p+r] #+ MA
        mu[i] = MU + MA
        resid[i] = rets[i] - mu[i]
        sigma[i] = np.sqrt(omega + alpha*resid[i-1]**2 + beta*sigma[i-1]**2)
        #mu[i]= MU + MA 
        MA = 0 
        MU = 0 
    #print(sigma, resid)
    return sigma**2, mu
            

def fMinusLoglikARMAGARCH(param,rets):    
    #print(param)
    vSigma2, mu = calcArma(param,rets)
    vLogPdf = -0.5*np.log(2*np.pi*vSigma2) -0.5*((rets- mu)/np.sqrt(vSigma2)) **2
    dMinusLogLikelihood = - np.sum(vLogPdf)
    #print(dMinusLogLikelihood)
    #print(dMinusLogLikelihood)
    return dMinusLogLikelihood


from numdifftools import Hessian

def fnOptim(p,q, bnds):
    paramAR0 = np.ones(p) - 0.9 #np.array([0.4, 0.2, 0.2])
    paramMA0 = np.ones(q) - 0.9 #np.array([0.1, 0.2, 0.4, 0.3])
    paramGARCH = np.array([0.2, 0.2, 0.3])
    #P = len(paramAR0)
    #Q = len(paramMA0)

    param0 = np.array([paramAR0, paramMA0, paramGARCH])
    param0 = np.concatenate(param0)
    #print(param0)
    #param0 = [0.4,0.3,0.1,0.2, 0.4,0.2, 0.2, 0.2,0.2,0.3]
    #print(param0)
    result = minimize(fMinusLoglikARMAGARCH, param0, args=np.array(y), method='slsqp', bounds= bnds ,options={'disp':True, 'maxiter':1000}) 
    hess = approx_hess(result.x, fMinusLoglikARMAGARCH, args = (np.array(y),))
    #hess = Hessian(fMinusLoglikARMAGARCH)(result.x, np.array(y))
    print('results : ',result.x)
    #print(Hessian)
    se = np.sqrt(np.diag(np.linalg.inv(hess)))
    print(se)
    ll = fMinusLoglikARMAGARCH(result.x, y) 
    n = len(y)
    k = (p + q + 3)
    AIC = 2*ll + 2*k 
    AICc = AIC + (2*k*(k+1))/(n-k-1)
    BIC = 2*ll + k*np.log(n)
    HQC = 2*ll + 2*k*np.log(np.log(n))
    print('AIC : ', AIC)
    print('AICc: ', AICc)
    print('BIC : ', BIC)
    print("HQC : ", HQC)
    return np.array([AIC, AICc, BIC, HQC])


P =  5
Q =  5
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(0, 10), (0,1), (0,1))
ARMA55GARCH = fnOptim(P, Q, bnds)

P =  5
Q =  4
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(0, 10), (0,1), (0,1))
ARMA54GARCH = fnOptim(P, Q, bnds)

P =  4
Q =  5
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(0, 10), (0,1), (0,1))
ARMA45GARCH = fnOptim(P, Q, bnds)

P =  4
Q =  4
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (0, 10), (0,1), (0,1))
ARMA44GARCH = fnOptim(P, Q, bnds)

P =  4
Q =  3
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (0, 10), (0,1), (0,1))
ARMA43GARCH = fnOptim(P, Q, bnds)

P = 3
Q = 4
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (0, 10), (0,1), (0,1))
ARMA34GARCH = fnOptim(P, Q, bnds)

P =  3
Q =  3
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1),(0, 10), (0,1), (0,1))
ARMA33GARCH = fnOptim(P, Q, bnds)

P =  3
Q =  1
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (0, 10), (0,1), (0,1))
ARMA31GARCH = fnOptim(P, Q, bnds)

P =  1
Q =  3
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (0, 10), (0,1), (0,1))
ARMA13GARCH = fnOptim(P, Q, bnds)

P =  2
Q =  3
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1),(0, 10), (0,1), (0,1))
ARMA23GARCH = fnOptim(P, Q, bnds)


P =  3
Q =  2
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (-1,1),(0, 10), (0,1), (0,1))
ARMA32GARCH = fnOptim(P, Q, bnds)

P =  2
Q =  2
bnds = ((-10,10), (-1,1), (-1,1), (-1,1), (0, 10), (0,1), (0,1))
ARMA22GARCH = fnOptim(P, Q, bnds)

P =  1
Q =  2
bnds = ((-10,10), (-1,1), (-1,1), (0, 10), (0,1), (0,1))
ARMA12GARCH = fnOptim(P, Q, bnds)

P =  2
Q =  1
bnds = ((-10,10), (-1,1), (-1,1), (0, 10), (0,1), (0,1))
ARMA21GARCH = fnOptim(P, Q, bnds)

P =  1
Q =  1
bnds = ((-10,10), (-1,1),(0, 10), (0,1), (0,1))
ARMA11GARCH = fnOptim(P, Q, bnds)

rowNames = ['ARMA11GARCH', 'ARMA21GARCH', 'ARMA12GARCH', 'ARMA22GARCH', 'ARMA31GARCH', 'ARMA13GARCH', 'ARMA32GARCH', 'ARMA23GARCH', 'ARMA33GARCH', "ARMA34GARCH",'ARMA43GARCH', 'ARMA44GARCH', "ARMA45GARCH",'ARMA54GARCH', 'ARMA55GARCH']
dataframe = pd.DataFrame(columns=["AIC", "AICc", "BIC", "HQC"], data=[ARMA11GARCH, ARMA21GARCH, ARMA12GARCH, ARMA22GARCH, ARMA31GARCH, ARMA13GARCH, ARMA32GARCH, ARMA23GARCH, ARMA33GARCH, ARMA34GARCH, ARMA43GARCH, ARMA44GARCH, ARMA45GARCH, ARMA54GARCH, ARMA55GARCH], index=rowNames).round(4)
print(dataframe.to_latex())