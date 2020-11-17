import pandas as pd # data frame library
import sys
import scipy as sc
import datetime # library for date-time functionality
import numpy as np # numerical python
from scipy import stats # scientific python statistical package
from scipy.optimize import curve_fit # optimization for fitting curves

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

#import warnings
#warnings.filterwarnings('ignore')

#### FUNCTIONS FOR MODELING THE SPREAD OF COVID-19 CASES

def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)


def gaussian1(x, n, s, m):  
    #return n**2 * (1/(s*((2*pi)**0.5))) * np.exp(-0.5 * ((x - m)/s)**2)
    return n**2 * 0.5 * (1 + sc.special.erf((x - m)/(s*2**0.5)))
    
def gaussian2(x, n, s, m, s1, m1):
    return n**2 * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5))))
    
def gaussian3(x, n, s, m, s1, m1, s2, m2):  
    return n**2 * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5))) + (1 + sc.special.erf((x - m2)/(s2*2**0.5))))
    
    
def phase_wave1(x,  a, b, c, d, f, g):
    return  a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))
    
def phase_wave2(x,  a, b, c, d, f, g,   a1, b1, c1, d1, g1, f1):
    return  a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))   +   a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1))
    
def phase_wave3(x, a, b, c, d, f, g,   a1, b1, c1, d1, g1, f1,   a2, b2, c2, d2, g2, f2):
        y = a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))
        y = y + a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1))
        y = y + a2 / (d2 + np.exp(-c2 * (x + g2*np.sin(f2*x)) + b2))
        #return  a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))   +   a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1))   +   a2 / (d2 + np.exp(-c2 * (x + g2*np.sin(f2*x)) + b2))
        return y
        
def logistic1(x, a, b, c, d):
    return  (a / (d + np.exp(-c * x + b)))
    
def logistic2(x, a, b, c, d, a1, b1, c1, d1):
    return  (a / (d + np.exp(-c * x + b)) + a1 / (d1 + np.exp(-c1 * x + b1)))
    
def logistic3(x, a, b, c, d,  a1, b1, c1, d1,  a2, b2, c2, d2):
    return  (a / (d + np.exp(-c * x + b)))   +   (a1 / (d1 + np.exp(-c1 * x + b1)))   +   (a2 / (d2 + np.exp(-c2 * x + b2)))
    

def most_likely(y0, n1, n2, r = 8):
    c = 10**-r
    wts = (c/(c + np.abs(y0 - n1))) ** r
    exp_y = np.average(n2, weights=wts)
    return exp_y


def WAF(obs_y, ForecastDays):
    
        c = 1
        obs_y = np.array(obs_y) + 1
        l = len(obs_y)
        
        ##### GET PREDICTED ######
        # obs_y is a cumulative list
        
        # n1 is a list holding the number of new cases reported for each day in the time series.
        # We assume the number of new cases reported on the first day is equal to the first 
        # value in obs_y, since nothing was reported before the first day.
        
        n0 = [obs_y[0]]
        for i, val in enumerate(obs_y):
            if i > 0:
                n0.append(obs_y[i] - obs_y[i-1])
        
        
        xx = list(range(len(n0))) #np.linspace(n0.min(), n0.max(), len(n0))
        # interpolate + smooth
        itp = interp1d(xx, n0, kind='linear')
        window_size, poly_order = 21, 2
        n0 = savgol_filter(itp(xx), window_size, poly_order)
        
        
        n1 = [0]
        for i, val in enumerate(n0):
            if i > 0:
                
                if n0[i] == 0:
                    n0[i] = c
                if n0[i-1] == 0:
                    n0[i-1] = c
                    
                n1.append((n0[i] - n0[i-1])/n0[i-1])

        
        # n2 is a staggered copy of n1. So, if the first element of n1 is the number of new cases
        # reported on the first day, then the first element of n2 is the number of new cases
        # reported on the second day.
            
        n2 = []
        l = len(n1) - 1
        for i, val in enumerate(n1):
            if i < l:
                n2.append(n1[i+1])
        
        # The last element of n1 has to be removed because there is no compliment to it in n2.
        n1 = n1[:-1]
        n1 = np.array(n1)
        n2 = np.array(n2)
        
        
        z = 1
        # Initiate the list of predicted values with the first value in n1. 
        pred_y = [obs_y[1]]
        for i, y1 in enumerate(n0):
            
            if i > 0:
                if n0[i] == 0:
                    n0[i] = c
                if n0[i-1] == 0:
                    n0[i-1] == c
                
                pc = (n0[i] - n0[i-1])/(n0[i-1])
                pc = most_likely(pc, n1, n2)
                
                y2 = y1 + pc*y1
                
                if y2 < 0:
                    y2 = 0
                pred_y.append(y2)
        
        #print(len(n0), len(n1), len(n2), len(pred_y), len(obs_y), len(obs_x))
        
        ##### GET FORECASTED ######
        
        forecasted_y = [pred_y[-2], pred_y[-1]]
        for i in range(ForecastDays + 1 - 2):
            
            if i > 0:
                if forecasted_y[i] == 0:
                    forecasted_y[i] = c
                if forecasted_y[i-1] == 0:
                    forecasted_y[i-1] == c
                
                pc = (forecasted_y[i] - forecasted_y[i-1])/(forecasted_y[i-1])
                pc = most_likely(pc, n1, n2)
                
                y2 = forecasted_y[i] + pc*forecasted_y[i]
                
                if y2 < 0:
                    y2 = 0
                forecasted_y.append(y2)
            
            
        forecasted_y = pred_y + forecasted_y[z:]
        return pred_y, forecasted_y




def opt_fit(obs_y, obs_x, forecasted_y, ForecastDays, model):
    
    r2_opt = 0
    ct = 0
    i = 3
    popt_opt = 0
    o_y = np.array(obs_y)
    o_y[-1] = o_y[-1] - ((o_y[-1] - o_y[-2]) * 0.5)

        
    while max(forecasted_y) > i * max(obs_y):
        ct += 1
        if ct > 10:
            break
        
        if model == 'WAF':
            pred_y, forecasted_y = WAF(o_y, ForecastDays)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            
        if model == 'phase_wave3':
            popt, pcov = curve_fit(phase_wave3, obs_x, o_y, sigma= 1 - 1/o_y,
                                       absolute_sigma=True, method='lm', maxfev=40000)
            pred_y = phase_wave3(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = phase_wave3(forecasted_x, *popt)
                
        elif model == 'phase_wave2':
            popt, pcov = curve_fit(phase_wave2, obs_x, o_y, sigma= 1 - 1/o_y,
                                       absolute_sigma=True, method='lm', maxfev=40000)
            pred_y = phase_wave2(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = phase_wave2(forecasted_x, *popt)
                
        elif model == 'logistic3':
            popt, pcov = curve_fit(logistic3, obs_x, o_y, sigma= 1 - 1/o_y,
                                       absolute_sigma=True, method='lm', maxfev=40000)
            pred_y = logistic3(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = logistic3(forecasted_x, *popt)
                
        elif model == 'logistic2':
            popt, pcov = curve_fit(logistic2, obs_x, o_y, sigma= 1 - 1/o_y,
                                       absolute_sigma=True, method='lm', maxfev=40000)
            pred_y = logistic2(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = logistic2(forecasted_x, *popt)
            
        r2 = obs_pred_rsquare(obs_y, pred_y)
        if r2 > r2_opt:
            r2_opt = float(r2)
            popt_opt = popt
            
        o_y[-1] = o_y[-1] - ((o_y[-1] - o_y[-2]) * 0.5)
    
    
    if model == 'WAF':
        pred_y, forecasted_y = WAF(o_y, ForecastDays)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        
    elif model == 'phase_wave3':
        pred_y = phase_wave3(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = phase_wave3(forecasted_x, *popt_opt)
                
    elif model == 'phase_wave2':
        pred_y = phase_wave2(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = phase_wave2(forecasted_x, *popt_opt)
                
    elif model == 'logistic3':
        pred_y = logistic3(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = logistic3(forecasted_x, *popt_opt)
                
    elif model == 'logistic2':
        pred_y = logistic2(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = logistic2(forecasted_x, *popt_opt)
            
    return pred_y, forecasted_x, forecasted_y





def get_WAF(obs_x, obs_y, ForecastDays):
    
    forecasted_y = [np.inf]
    o_y = np.array(obs_y)
    
    pred_y, forecasted_y = WAF(o_y, ForecastDays)
    pred_y = [sum(pred_y[0:x:1]) for x in range(len(pred_y)+1)][:-1]
    forecasted_y = [sum(forecasted_y[0:x:1]) for x in range(len(forecasted_y)+1)][:-1]
    
    r2 = obs_pred_rsquare(obs_y, pred_y)
    if r2 in [np.inf, -np.inf, np.nan]:
        g = 1 + []
            
    elif max(forecasted_y) > 2*max(obs_y):
        model = 'WAF'
        #o_y[-i:] = o_y[-i:] - ((o_y[-i:] - o_y[-i-1]) * 0.001)
        
        pred_y, forecasted_x, forecasted_y = opt_fit(obs_y, obs_x, forecasted_y, ForecastDays, model)
        pred_y = [sum(pred_y[0:x:1]) for x in range(len(pred_y)+1)][:-1]
        forecasted_y = [sum(forecasted_y[0:x:1]) for x in range(len(forecasted_y)+1)][:-1]
        
    forecasted_x = list(range(len(forecasted_y)))
    params = []

    return forecasted_y, forecasted_x, pred_y, params





def get_gaussian(obs_x, obs_y, ForecastDays):
    
    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x) 
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    
    # convert obs_y to numpy array
    obs_y = np.array(obs_y)
    
    try:
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(gaussian3, obs_x, obs_y,
                               method='lm', maxfev=20000)
        # get predicted y values
        pred_y = gaussian3(obs_x, *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = gaussian3(forecasted_x, *popt)
        
        r2 = obs_pred_rsquare(obs_y, pred_y)
        if r2 in [np.inf, -np.inf, np.nan]:
            g = 1 + []
            
    except:
        try:
            
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(gaussian2, obs_x, obs_y,
                                   method='lm', maxfev=20000)
            # get predicted y values
            pred_y = gaussian2(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = gaussian2(forecasted_x, *popt)
            
            r2 = obs_pred_rsquare(obs_y, pred_y)
            if r2 in [np.inf, -np.inf, np.nan]:
                g = 1 + []
                
        except:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(gaussian1, obs_x, obs_y,
                                   method='lm', maxfev=20000)
            # get predicted y values
            pred_y = gaussian1(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = gaussian1(forecasted_x, *popt)
    
    
    # return the forecasted x-y values and predicted y values
    params = []
    
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            forecasted_y[i] = 5*max(pred_y)
            
        elif i > 0 and val <= 0 and forecasted_y[i-1] > 0:
            forecasted_y[i] = max(pred_y)
            
        if i > 0 and forecasted_y[i] < forecasted_y[i-1]:
            forecasted_y[i] = forecasted_y[i-1]
            
    return forecasted_y, forecasted_x, pred_y, params




def get_phase_wave(obs_x, obs_y, ForecastDays):
    
        # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x) 
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    
    # convert obs_y to numpy array
    obs_y = np.array(obs_y)
    
    try:
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(phase_wave3, obs_x, obs_y,
                               method='lm', maxfev=20000)
        # get predicted y values
        pred_y = phase_wave3(obs_x, *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = phase_wave3(forecasted_x, *popt)
        
        r2 = obs_pred_rsquare(obs_y, pred_y)
        if r2 in [np.inf, -np.inf, np.nan]:
            g = 1 + []
            
        elif max(forecasted_y) > 3*max(obs_y):
            model = 'phase_wave3'
            pred_y, forecasted_x, forecasted_y = opt_fit(obs_y, obs_x, forecasted_y, ForecastDays, model)
            
    except:
        try:
            
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(phase_wave2, obs_x, obs_y,
                                   method='lm', maxfev=20000)
            # get predicted y values
            pred_y = phase_wave2(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = phase_wave2(forecasted_x, *popt)
            
            r2 = obs_pred_rsquare(obs_y, pred_y)
            if r2 in [np.inf, -np.inf, np.nan]:
                g = 1 + []
                
            elif max(forecasted_y) > 3*max(obs_y):
                model = 'phase_wave2'
                pred_y, forecasted_x, forecasted_y = opt_fit(obs_y, obs_x, forecasted_y, ForecastDays, model)
            
                
        except:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(phase_wave1, obs_x, obs_y,
                                   method='lm', maxfev=20000)
            # get predicted y values
            pred_y = phase_wave1(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = phase_wave1(forecasted_x, *popt)
    
    
    # return the forecasted x-y values and predicted y values
    params = []
    
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            forecasted_y[i] = 5*max(pred_y)
            
        elif i > 0 and val <= 0 and forecasted_y[i-1] > 0:
            forecasted_y[i] = max(pred_y)
            
        if i > 0 and forecasted_y[i] < forecasted_y[i-1]:
            forecasted_y[i] = forecasted_y[i-1]
            
    return forecasted_y, forecasted_x, pred_y, params







def get_logistic(obs_x, obs_y, ForecastDays):
    
        # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x) 
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    
    # convert obs_y to numpy array
    obs_y = np.array(obs_y)
    
    try:
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(logistic3, obs_x, obs_y,
                               method='lm', maxfev=20000)
        # get predicted y values
        pred_y = logistic3(obs_x, *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = logistic3(forecasted_x, *popt)
        
        r2 = obs_pred_rsquare(obs_y, pred_y)
        if r2 in [np.inf, -np.inf, np.nan]:
            g = 1 + []
            
        elif max(forecasted_y) > 3*max(obs_y):
            model = 'logistic3'
            pred_y, forecasted_x, forecasted_y = opt_fit(obs_y, obs_x, forecasted_y, ForecastDays, model)
        
            
    except:
        try:
            
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(logistic2, obs_x, obs_y,
                                   method='lm', maxfev=20000)
            # get predicted y values
            pred_y = logistic2(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = logistic2(forecasted_x, *popt)
            
            r2 = obs_pred_rsquare(obs_y, pred_y)
            if r2 in [np.inf, -np.inf, np.nan]:
                g = 1 + []
                
            elif max(forecasted_y) > 3*max(obs_y):
                model = 'logistic2'
                pred_y, forecasted_x, forecasted_y = opt_fit(obs_y, obs_x, forecasted_y, ForecastDays, model)
            
                
        except:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(logistic1, obs_x, obs_y,
                                   method='lm', maxfev=20000)
            # get predicted y values
            pred_y = logistic1(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = logistic1(forecasted_x, *popt)
    
    
    # return the forecasted x-y values and predicted y values
    params = []
    
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            forecasted_y[i] = 5*max(pred_y)
            
        elif i > 0 and val <= 0 and forecasted_y[i-1] > 0:
            forecasted_y[i] = max(pred_y)
            
        if i > 0 and forecasted_y[i] < forecasted_y[i-1]:
            forecasted_y[i] = forecasted_y[i-1]
            
    return forecasted_y, forecasted_x, pred_y, params





def get_exponential(obs_x, obs_y, ForecastDays):
    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x) 
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass
    # convert obs_y to numpy array
    obs_y = obs_y.tolist()      
    
    # use linear regression to obtain the estimated parameters for
    # an exponential fit. This is not an optimization procedure,
    # but should not fail to provide a best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(obs_x, np.log(obs_y))
    # convert obs_y to numpy array
    obs_y = np.array(obs_y)
    
    # get predicted y values
    pred_y = np.exp(intercept + slope*obs_x)
    # extend x values by number of ForecastDays
    forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
    # get corresponding forecasted y values, i.e., extend the predictions
    forecasted_y = np.exp(intercept + slope*forecasted_x)
    
    # return the forecasted x-y values and predicted y values
    params = []
    
    obs_x = 0
    obs_y = 0
    
    fy = []
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            fy.append(5*max(pred_y))
            
        else:
            fy.append(val)
            
            
    return fy, forecasted_x, pred_y, params
        


def get_polynomial(obs_x, obs_y, ForecastDays, degree=2):
    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x)
    
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass       
    
    # convert obs_y to numpy array
    obs_y = obs_y.tolist()
    
    try:
        # attempt to fit the polynomial model to the observed data
        # z: best-fit model parameter values
        z = np.polyfit(obs_x, obs_y, degree)
        # p: one-dimensional polynomial class; constructs the polynomial
        p = np.poly1d(z)
        # get predicted y values
        pred_y = p(obs_x)
        
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = p(forecasted_x)
    except:
        # if the polynomial model fails, the lack of a substitute here
        # will throw an error
        pass
    
    params = []
    obs_x = 0
    obs_y = 0
    
    fy = []
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            fy.append(5*max(pred_y))
            
        else:
            fy.append(val)
            
    # return the forecasted x-y values and predicted y values
    return fy, forecasted_x, pred_y, params





def get_2phase_logistic(obs_x, obs_y, ForecastDays):
    
    def logistic(x, a, b, c, d):
        # A general logistic function
        # x is observed data
        # a, b, c are optimized by scipy optimize curve fit
        return a / (d + np.exp(-c * x + b))
        

    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass
    
    try:
        
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(logistic, 
                               np.float64(obs_x), 
                               np.float64(obs_y), 
                               method='lm', maxfev=2000)
        
        # get predicted y values
        if np.isinf(pcov[0][0]) == True:
            check = 'check' + pcov[0][0]
        
        pred_y = logistic(np.float64(obs_x), *popt)
        # extend x values by number of ForecastDays
        if ForecastDays > 0:
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = logistic(np.float64(forecasted_x), *popt)
        
        else:
            forecasted_y = np.copy(pred_y)
            forecasted_x = np.copy(obs_x)
            
                        
        
        
    except:
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(logistic, 
                               np.float64(obs_x), 
                               np.float64(obs_y), 
                               method='lm', maxfev=2000)
        # get predicted y values
        pred_y = logistic(np.float64(obs_x), *popt)
        # extend x values by number of ForecastDays
        if ForecastDays > 0:
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = logistic(np.float64(forecasted_x), *popt)
        
        else:
            forecasted_y = np.copy(pred_y)
            forecasted_x = np.copy(obs_x)
    
    params = []
    fy = []
    for i, val in enumerate(forecasted_y):
        if val > 10*max(pred_y):
            fy.append(10*max(pred_y))
            
        elif val < 1:
            fy.append(10*max(pred_y))
            
        else:
            fy.append(val)
    
    # prevent use of negative y values and
    # trailing zero-valued y values
    #for i, val in enumerate(fy):
    #    if val < 1:
    #        try:
    #            obs_y[i] = obs_y[i-1]
    #        except:
    #            pass
                    
    fy = np.array(fy)
    return fy, forecasted_x, pred_y, params





def get_sine_logistic(obs_x, obs_y, ForecastDays):
    
    def lapse_logistic(x, a, b, c, d):
        # A general logistic function
        # x is observed data
        # a, b, c are optimized by scipy optimize curve fit
        return a / (d + np.exp(-c * x + b))
    
    
    def sine_logistic(x, a, b, c, d, f, g):
        # A general logistic function
        # x is observed data
        # a, b, c are optimized by scipy optimize curve fit
        return a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))
        

    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass
    
    try:
        
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(sine_logistic, 
                               np.float64(obs_x), 
                               np.float64(obs_y), 
                               method='lm', maxfev=2000)
        
        # get predicted y values
        if np.isinf(pcov[0][0]) == True:
            check = 'check' + pcov[0][0]
        
        pred_y = sine_logistic(np.float64(obs_x), *popt)
        # extend x values by number of ForecastDays
        if ForecastDays > 0:
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = sine_logistic(np.float64(forecasted_x), *popt)
        
        else:
            forecasted_y = np.copy(pred_y)
            forecasted_x = np.copy(obs_x)
            
                        
        
        
    except:
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(lapse_logistic, 
                               np.float64(obs_x), 
                               np.float64(obs_y), 
                               method='lm', maxfev=2000)
        # get predicted y values
        pred_y = lapse_logistic(np.float64(obs_x), *popt)
        # extend x values by number of ForecastDays
        if ForecastDays > 0:
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = lapse_logistic(np.float64(forecasted_x), *popt)
        
        else:
            forecasted_y = np.copy(pred_y)
            forecasted_x = np.copy(obs_x)
    
    params = []
    fy = []
    for i, val in enumerate(forecasted_y):
        if val > 10*max(pred_y):
            fy.append(10*max(pred_y))
            
        elif val < 1:
            fy.append(10*max(pred_y))
            
        else:
            fy.append(val)
    
    # prevent use of negative y values and
    # trailing zero-valued y values
    #for i, val in enumerate(fy):
    #    if val < 1:
    #        try:
    #            obs_y[i] = obs_y[i-1]
    #        except:
    #            pass
                    
    fy = np.array(fy)
    return fy, forecasted_x, pred_y, params






def fit_curve(condition):
    
    obs_x, obs_y, model, ForecastDays, day, iterations = condition
    # A function to fit various models to observed COVID-19 cases data according to:
    # obs_x: observed x values
    # obs_y: observed y values
    # model: the model to fit
    # ForecastDays: number of days ahead to extend predictions
    # N: population size of interest

    
    # use the number of y observations as the number of x observations
    obs_x = list(range(len(obs_y)))
    # convert y and x observations to numpy arrays
    obs_x = np.array(obs_x)
    obs_y = np.array(obs_y)
    
    
    # Get the forecasted values, predicted values, and observed vs predicted r-square
    # value for the chosen model
    
    if model == '2 phase sine-logistic':
        
        max_r2 = 0
        b_pt = 90
        for i in range(60, 130):
        
            obs_x1 = obs_x[0:i]
            obs_y1 = obs_y[0:i]
        
            obs_x2 = obs_x[i:]
            obs_y2 = obs_y[i:]
            
            
            miny = max(obs_y1) * 0.7    
            obs_y2 = np.array(obs_y2) - miny
            
            minx = min(obs_x2)
            obs_x2 = np.array(obs_x2) - minx
            
            
            forecasted_y1, forecasted_x1, pred_y1, params1 = get_sine_logistic(obs_x1, obs_y1, 0)
            forecasted_y2, forecasted_x2, pred_y2, params2 = get_sine_logistic(obs_x2, obs_y2, ForecastDays)
            
            obs_y2 = np.array(obs_y2) + miny
            forecasted_y2 = np.array(forecasted_y2) + miny
            pred_y2 = np.array(pred_y2) + miny
            
            obs_x2 = np.array(obs_x2) + minx
            forecasted_x2 = np.array(forecasted_x2) + minx
    
            
            forecasted_x = forecasted_x1.tolist() + forecasted_x2.tolist()
            forecasted_y = forecasted_y1.tolist() + forecasted_y2.tolist()
            pred_y = pred_y1.tolist() + pred_y2.tolist()
            obs_y = obs_y1.tolist() + obs_y2.tolist()
            
            pred_y = np.array(pred_y)
            obs_y = np.array(obs_y)
            forecasted_x = np.array(forecasted_x)
            forecasted_y = np.array(forecasted_y)
            
            obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
            if obs_pred_r2 > max_r2:
                max_r2 = obs_pred_r2
                b_pt = i
        
        obs_x1 = obs_x[0:b_pt]
        obs_y1 = obs_y[0:b_pt]
        
        obs_x2 = obs_x[b_pt:]
        obs_y2 = obs_y[b_pt:]
            
            
        miny = max(obs_y1) * 0.7    
        obs_y2 = np.array(obs_y2) - miny
            
        minx = min(obs_x2)
        obs_x2 = np.array(obs_x2) - minx
            
            
        forecasted_y1, forecasted_x1, pred_y1, params1 = get_sine_logistic(obs_x1, obs_y1, 0)
        forecasted_y2, forecasted_x2, pred_y2, params2 = get_sine_logistic(obs_x2, obs_y2, ForecastDays)
            
        obs_y2 = np.array(obs_y2) + miny
        forecasted_y2 = np.array(forecasted_y2) + miny
        pred_y2 = np.array(pred_y2) + miny
            
        obs_x2 = np.array(obs_x2) + minx
        forecasted_x2 = np.array(forecasted_x2) + minx
    
            
        forecasted_x = forecasted_x1.tolist() + forecasted_x2.tolist()
        forecasted_y = forecasted_y1.tolist() + forecasted_y2.tolist()
        pred_y = pred_y1.tolist() + pred_y2.tolist()
        obs_y = obs_y1.tolist() + obs_y2.tolist()
            
        pred_y = np.array(pred_y)
        obs_y = np.array(obs_y)
        forecasted_x = np.array(forecasted_x)
        forecasted_y = np.array(forecasted_y)
            
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
            
        #print('\n')
        #print(obs_y)
        #print(pred_y)
        #print(obs_pred_r2)
        
        params = params1.extend(params2)
        
    
    elif model == '2 phase logistic':
        
        max_r2 = 0
        b_pt = 90
        for i in range(60, 130):
        
            obs_x1 = obs_x[0:i]
            obs_y1 = obs_y[0:i]
        
            obs_x2 = obs_x[i:]
            obs_y2 = obs_y[i:]
            
            
            miny = max(obs_y1) * 0.7    
            obs_y2 = np.array(obs_y2) - miny
            
            minx = min(obs_x2)
            obs_x2 = np.array(obs_x2) - minx
            
            
            forecasted_y1, forecasted_x1, pred_y1, params1 = get_2phase_logistic(obs_x1, obs_y1, 0)
            forecasted_y2, forecasted_x2, pred_y2, params2 = get_2phase_logistic(obs_x2, obs_y2, ForecastDays)
            
            obs_y2 = np.array(obs_y2) + miny
            forecasted_y2 = np.array(forecasted_y2) + miny
            pred_y2 = np.array(pred_y2) + miny
            
            obs_x2 = np.array(obs_x2) + minx
            forecasted_x2 = np.array(forecasted_x2) + minx
    
            
            forecasted_x = forecasted_x1.tolist() + forecasted_x2.tolist()
            forecasted_y = forecasted_y1.tolist() + forecasted_y2.tolist()
            pred_y = pred_y1.tolist() + pred_y2.tolist()
            obs_y = obs_y1.tolist() + obs_y2.tolist()
            
            pred_y = np.array(pred_y)
            obs_y = np.array(obs_y)
            forecasted_x = np.array(forecasted_x)
            forecasted_y = np.array(forecasted_y)
            
            obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
            if obs_pred_r2 > max_r2:
                max_r2 = obs_pred_r2
                b_pt = i
        
        obs_x1 = obs_x[0:b_pt]
        obs_y1 = obs_y[0:b_pt]
        
        obs_x2 = obs_x[b_pt:]
        obs_y2 = obs_y[b_pt:]
            
            
        miny = max(obs_y1) * 0.7    
        obs_y2 = np.array(obs_y2) - miny
            
        minx = min(obs_x2)
        obs_x2 = np.array(obs_x2) - minx
            
            
        forecasted_y1, forecasted_x1, pred_y1, params1 = get_logistic(obs_x1, obs_y1, 0)
        forecasted_y2, forecasted_x2, pred_y2, params2 = get_logistic(obs_x2, obs_y2, ForecastDays)
            
        obs_y2 = np.array(obs_y2) + miny
        forecasted_y2 = np.array(forecasted_y2) + miny
        pred_y2 = np.array(pred_y2) + miny
            
        obs_x2 = np.array(obs_x2) + minx
        forecasted_x2 = np.array(forecasted_x2) + minx
    
            
        forecasted_x = forecasted_x1.tolist() + forecasted_x2.tolist()
        forecasted_y = forecasted_y1.tolist() + forecasted_y2.tolist()
        pred_y = pred_y1.tolist() + pred_y2.tolist()
        obs_y = obs_y1.tolist() + obs_y2.tolist()
            
        pred_y = np.array(pred_y)
        obs_y = np.array(obs_y)
        forecasted_x = np.array(forecasted_x)
        forecasted_y = np.array(forecasted_y)
            
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
            
        
        params = params1.extend(params2)
        
        
        
    if model == 'Phase Wave':
        forecasted_y, forecasted_x, pred_y, params = get_phase_wave(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
    
    elif model == 'WAF':
        forecasted_y, forecasted_x, pred_y, params = get_WAF(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
        
    elif model == 'Logistic (multi-phase)':
        forecasted_y, forecasted_x, pred_y, params = get_logistic(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
        
    elif model == 'Gaussian (multi-phase)':
        forecasted_y, forecasted_x, pred_y, params = get_gaussian(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
    
    elif model == 'Exponential':
        forecasted_y, forecasted_x, pred_y, params = get_exponential(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
    
    elif model == 'Quadratic':
        forecasted_y, forecasted_x, pred_y, params = get_polynomial(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
        
    obs_y = 0
    
    return [obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params]
