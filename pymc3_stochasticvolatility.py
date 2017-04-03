# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:03:57 2017

@author: dahoang
"""

# getting started
# install theano
#   http://deeplearning.net/software/theano/install_windows.html#install-requirements-and-optional-packages
#   http://stackoverflow.com/questions/33687103/how-to-install-theano-on-anaconda-python-2-7-x64-on-windows
# install pymc3
#   API: https://pymc-devs.github.io/pymc3/api.html
# on AIG work (physical) desktop, takes 5-6 min to run!
# other resources:
#   http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
#   http://twiecki.github.io/blog/2016/06/01/bayesian-deep-learning/
#   https://pandas-datareader.readthedocs.io/en/latest/remote_data.html

import matplotlib.pyplot as plt
import numpy as np

from pymc3 import Model
from pymc3 import Exponential, StudentT, Deterministic
from pymc3.distributions.timeseries import GaussianRandomWalk
from pymc3.math import exp

from pymc3 import variational

from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot

# get the data!
try:
    from pandas_datareader import data
except ImportError:
    #!pip install pandas-datareader
    import pip
    pip.main(['install', 'pandas-datareader']) 
    #pip.main(['install', '--user', 'pandas-datareader'])
    from pandas_datareader import data
returns = data.get_data_yahoo('SPY', start='2008-5-1', end='2009-12-1')['Adj Close'].pct_change()
#print(len(returns))

# plot the pulled data!
returns.plot()#figsize=(10, 6))
plt.ylabel('daily returns in %');

# define the model
# \sig ~ exp(50)
#       why? stdev of returns is approx 0.02
#       stdev of exp(lam=50) = 0.2
# \nu ~ exp(0.1)
#       the DOF for the student T...which should be sample size
#       mean of exp(lam=0.1) = 10
# s_i ~ normal(s_i-1, \sig^-2)
# log(y_i) ~ studentT(\nu, 0, exp(-2s_i))
with Model() as sp500_model:
    nu = Exponential('nu', 1./10, testval=5.)#50, testval=5.)#results similar...
    sigma = Exponential('sigma', 1./.02, testval=.1)
    s = GaussianRandomWalk('s', sigma**-2, shape=len(returns))
    volatility_process = Deterministic('volatility_process', exp(-2*s))
    r = StudentT('r', nu, lam=1/volatility_process, observed=returns)


# fit the model using NUTS
# NUTS is auto-assigned in sample()...why?
# you may get an error like:
#   WARNING (theano.gof.compilelock): Overriding existing lock by dead process '10876' (I am process '3456')
# ignore it...the process will move along
with sp500_model:
    trace = sample(2000, progressbar=False)
# plot results from model fitting...
# is there a practical reason for starting the plot from 200th sample
traceplot(trace[200:], [nu, sigma]);


# plot the results: volatility inferred by the model
fig, ax = plt.subplots()#figsize=(15, 8))
returns.plot(ax=ax)
ax.plot(returns.index, 1/np.exp(trace['s',::5].T), 'r', alpha=.03);
ax.set(title='volatility_process', xlabel='time', ylabel='volatility');
ax.legend(['S&P500', 'stochastic volatility process'])