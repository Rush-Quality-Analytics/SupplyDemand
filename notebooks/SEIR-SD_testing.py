import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import datetime

import re
import warnings

import sys
from scipy.optimize import curve_fit
from scipy import stats

import os

import base64
import time

from random import choice, shuffle



    
    
def obs_pred_rsquare(obs, pred):
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)


def seir_sd(obs_x, obs_y, ForecastDays, init_vals, params, N, t, sd):
    # This model is derived from: https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296
    
    obs_x = np.array(obs_x)
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass       
    
    
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R, Ir = [S_0], [E_0], [I_0], [R_0], [I_0]
    alpha, beta, gamma, rho = params
    i = 1
    
    for j in t[1:]:
        
        I_N = I[-1]/N
        
        sd1 = sd/(sd + I_N)
        beta = beta * sd1
        
        next_S = ((S[-1] - (beta*S[-1]*I[-1])))
        next_E = (E[-1] + (beta*S[-1]*I[-1] - alpha * E[-1]))
        
        next_I = I[-1] + (alpha*E[-1] - gamma * I[-1])
        next_R = R[-1] + (gamma*I[-1]) 
        
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
        
        test_lag = 1/(1+np.exp(-0.1*j+5))
        Ir.append(next_I * test_lag)
        
                
    I = np.array(Ir)*N
    forecasted_y = I.tolist()
    
    num = len(obs_y)+ForecastDays
    pred_y = forecasted_y[-num+1:-ForecastDays+1]
    pred_y = np.array(pred_y)
    
    forecasted_y = forecasted_y[-num+1:]
    forecasted_y = np.array(forecasted_y)
    forecasted_x = np.array(range(len(forecasted_y)))
    
    print(len(obs_x), len(obs_y), len(pred_y))
    print(len(forecasted_x), len(forecasted_y))
    
    return forecasted_y, forecasted_x, pred_y









df = pd.read_csv('COVID-CASES-DF.txt', sep='\t')  
df = df[df['Country/Region'] == 'US']
df = df[df['Province/State'] != 'US']

focal_loc = 'Illinois'
#focal_loc = 'New York'
df_sub = df[df['Province/State'] == focal_loc]
df_sub = df_sub.loc[:, (df_sub != 0).any(axis=0)]

yi = list(df_sub)
focal = df_sub.iloc[0,4:].values

obs_y = []
dates = []
for ii, val in enumerate(focal):
    if len(obs_y) > 0 or val > 0:
        obs_y.append(val)

obs_x = list(range(len(obs_y)))



# Define parameters
ForecastDays = 30
ForecastDays += 1
t_max = 63 + ForecastDays

N = 12740000
#N = 8623000

#S, E, I, R
init_vals = 1 - 1/N, 1/N, 0, 0

incubation_period = 5
alpha = 1/incubation_period 

infectious_period = 7
gamma = 1/infectious_period 

rho = 4
beta = gamma*rho


sd =  0.0000000007
#sd = 0.00000001

params = alpha, beta, gamma, rho


t = list(range(t_max))


forecasted_y, forecasted_x, pred_y = seir_sd(obs_x, obs_y, ForecastDays,
                                        init_vals, params, N, t, sd)

obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)  
print('r2:', obs_pred_r2) 

fig = plt.figure()
plt.plot(obs_x, obs_y)  
plt.plot(forecasted_x, forecasted_y)        
