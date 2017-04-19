import numpy as np
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
challenger_data=np.genfromtxt("G:/量化投资/概率图模型/challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
challenger_data=challenger_data[~np.isnan(challenger_data[:,1])]

temperature =challenger_data[:,0]
D=challenger_data[:,1]
def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))
'''
获得后验分布，前验分布为正态分布
'''
with pm.Model() as model:
    beta=pm.Normal("beta",mu=0,tau=0.001,testval=0)
    alpha=pm.Normal("alpha",mu=0,tau=0.001,testval=0)
    p=pm.Deterministic("p",1.0 / (1.0 + tt.exp(beta*temperature + alpha)))
    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, start=start)
    burned_trace = trace[100000::2]

alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d 
beta_samples = burned_trace["beta"][:, None]
'''
预测
'''
t = np.linspace(temperature.min() - 5, temperature.max()+5, 51)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)
mean_prob_t = p_t.mean(axis=0)#跟定t的平均概率
qs = mquantiles(p_t, [0.025, 0.975], axis=0) #95%置信区间

'''
评价:看实际上的发生的情况，是否与 预测的概率分布相近:
低发生概率的部分(预测) 实际发生的概率小于高发生概率部分(预测)的发生概率（实际观察到的情况)
'''
temperaturetmp=temperature[:,None]
p_predict=logistic(temperaturetmp.T,beta_samples,alpha_samples).mean(axis=0)
result=np.array((temperature,D,p_predict)).T
resultdata=pd.DataFrame(result,columns=['temperatrue','AccidentObserved','Prediction'])
resultdata=resultdata.sort(columns='Prediction')
