import pandas as pd # data frame library
import sys
import scipy as sc
import datetime # library for date-time functionality
import numpy as np # numerical python
from scipy import stats # scientific python statistical package
from scipy.optimize import curve_fit # optimization for fitting curves

#### FUNCTIONS FOR MODELING THE SPREAD OF COVID-19 CASES

def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)

################ Simple growth-based statistical models


def get_gaussian(obs_x, obs_y, ForecastDays):
    
    def gaussian1(x, n, s, m):  
        #return n**2 * (1/(s*((2*pi)**0.5))) * np.exp(-0.5 * ((x - m)/s)**2)
        return n**2 * 0.5 * (1 + sc.special.erf((x - m)/(s*2**0.5)))
    
    def gaussian2(x, n, s, m, s1, m1):
        return n**2 * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5))))
    
    def gaussian3(x, n, s, m, s1, m1, s2, m2):  
        return n**2 * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5))) + (1 + sc.special.erf((x - m2)/(s2*2**0.5))))
    
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
    
    def phase_wave1(x,  a, b, c, d, f, g):
        return  a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))
    
    def phase_wave2(x,  a, b, c, d, f, g,   a1, b1, c1, d1, g1, f1):
        return  a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))   +   a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1))
    
    def phase_wave3(x,
                    a, b, c, d, f, g,   
                    a1, b1, c1, d1, g1, f1,   
                    a2, b2, c2, d2, g2, f2,
                    ):
        
        y = a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))
        y = y + a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1))
        y = y + a2 / (d2 + np.exp(-c2 * (x + g2*np.sin(f2*x)) + b2))
        
        return y
        
        #return  a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))   +   a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1))   +   a2 / (d2 + np.exp(-c2 * (x + g2*np.sin(f2*x)) + b2))
    
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
        obs_x = np.array(obs_x)
        obs_y = np.array(obs_y)
        forecasted_y = [np.inf]
        forecasted_x = [0]
        pred_y = [np.inf]
        r2_opt = 1
        popt_opt = 1
        try:
            popt, pcov = curve_fit(phase_wave3, 
                                   obs_x, 
                                   obs_y, 
                                   sigma= 1 - 1/obs_y,
                                   absolute_sigma=True,
                                   method='lm', 
                                   maxfev=40000)
                    
            pred_y = phase_wave3(obs_x, *popt)
                    
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = phase_wave3(forecasted_x, *popt)
        
        except:
            pass
        
        if max(forecasted_y) <= 3 * max(obs_y) and max(forecasted_y) >= max(obs_y):
            popt_opt = (popt)
            pass

        else:
            
            r2_opt = 0
            popt_opt = 0
            for i in [3]:
                    
                o_y = np.array(obs_y)
                ct = 0
                
                while max(forecasted_y) > i * max(obs_y):
                    ct += 1
                    if ct > 10:
                        break
                    
                    try:
                        popt, pcov = curve_fit(phase_wave3, 
                                               obs_x, 
                                               o_y, 
                                               sigma= 1 - 1/o_y,
                                               absolute_sigma=True,
                                               method='lm', 
                                               maxfev=40000)
                        
                        pred_y = phase_wave3(obs_x, *popt)
                        
                        # extend x values by number of ForecastDays
                        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
                        # get corresponding forecasted y values, i.e., extend the predictions
                        forecasted_y = phase_wave3(forecasted_x, *popt)
                        
                        if max(forecasted_y) > i * max(obs_y):
                            o_y[-1] = o_y[-1] - ((o_y[-1] - o_y[-2]) * 0.5)
                            #o_y[-1:] = o_y[-1:]**0.999
                        
                        else:
                            r2 = obs_pred_rsquare(obs_y, pred_y)
                            if r2 > r2_opt:
                                r2_opt = float(r2)
                                popt_opt = popt
                            
                                
                    except:
                        continue
        
        if r2_opt > 0:
            pred_y = phase_wave3(obs_x, *popt_opt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = phase_wave3(forecasted_x, *popt_opt)
                        
                
    except:
        
        try:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            obs_x = np.array(obs_x)
            obs_y = np.array(obs_y)
            forecasted_y = [np.inf]
            forecasted_x = [0]
            pred_y = [np.inf]
            r2_opt = 1
            popt_opt = 1
            try:
                popt, pcov = curve_fit(phase_wave2, 
                                       obs_x, 
                                       obs_y, 
                                       sigma= 1 - 1/obs_y,
                                       absolute_sigma=True,
                                       method='lm', 
                                       maxfev=40000)
                        
                pred_y = phase_wave2(obs_x, *popt)
                        
                # extend x values by number of ForecastDays
                forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
                # get corresponding forecasted y values, i.e., extend the predictions
                forecasted_y = phase_wave2(forecasted_x, *popt)
            
            except:
                pass
            
            if max(forecasted_y) <= 2 * max(obs_y) and max(forecasted_y) >= max(obs_y):
                popt_opt = (popt)
                pass
    
            else:
                
                r2_opt = 0
                popt_opt = 0
                for i in [2]:
                        
                    o_y = np.array(obs_y)
                    ct = 0
                    
                    while max(forecasted_y) > i * max(obs_y):
                        ct += 1
                        if ct > 10:
                            break
                        
                        try:
                            popt, pcov = curve_fit(phase_wave2, 
                                                   obs_x, 
                                                   o_y, 
                                                   sigma= 1 - 1/o_y,
                                                   absolute_sigma=True,
                                                   method='lm', 
                                                   maxfev=40000)
                            
                            pred_y = phase_wave2(obs_x, *popt)
                            
                            # extend x values by number of ForecastDays
                            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
                            # get corresponding forecasted y values, i.e., extend the predictions
                            forecasted_y = phase_wave2(forecasted_x, *popt)
                            
                            if max(forecasted_y) > i * max(obs_y):
                                
                                o_y[-1] = o_y[-1] - ((o_y[-1] - o_y[-2]) * 0.5)
                                #o_y[-1:] = o_y[-1:]**0.999
                            
                            else:
                                r2 = obs_pred_rsquare(obs_y, pred_y)
                                if r2 > r2_opt:
                                    r2_opt = float(r2)
                                    popt_opt = popt
                                
                                    
                        except:
                            continue
            
            if r2_opt > 0:
                pred_y = phase_wave2(obs_x, *popt_opt)
                forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
                forecasted_y = phase_wave2(forecasted_x, *popt_opt)
                            
            
                
        except:
            
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            obs_x = np.array(obs_x)
            obs_y = np.array(obs_y)
                
            o_y = np.array(obs_y)
            popt, pcov = curve_fit(phase_wave1, 
                                               obs_x, 
                                               o_y, 
                                               #sigma= 1 - 1/o_y,
                                               #absolute_sigma=True,
                                               method='lm', 
                                               maxfev=40000)
                        
            pred_y = phase_wave1(obs_x, *popt)
                        
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = phase_wave1(forecasted_x, *popt)
    
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
    
    
    def logistic1(x, a, b, c, d):
        return  (a / (d + np.exp(-c * x + b)))
    
    def logistic2(x, a, b, c, d, a1, b1, c1, d1):
        return  (a / (d + np.exp(-c * x + b)) + a1 / (d1 + np.exp(-c1 * x + b1)))
    
    def logistic3(x, a, b, c, d,  a1, b1, c1, d1,  a2, b2, c2, d2):
        return  (a / (d + np.exp(-c * x + b)))   +   (a1 / (d1 + np.exp(-c1 * x + b1)))   +   (a2 / (d2 + np.exp(-c2 * x + b2)))
    
    
    
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
        obs_x = np.array(obs_x)
        obs_y = np.array(obs_y)
        forecasted_y = [np.inf]
        forecasted_x = [0]
        pred_y = [np.inf]
        r2_opt = 1
        popt_opt = 1
        try:
            popt, pcov = curve_fit(logistic3, 
                                   obs_x, 
                                   obs_y, 
                                   sigma= 1 - 1/obs_y,
                                   absolute_sigma=True,
                                   method='lm', 
                                   maxfev=40000)
                    
            pred_y = logistic3(obs_x, *popt)
                    
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = logistic3(forecasted_x, *popt)
        
        except:
            pass
        
        if max(forecasted_y) <= 3 * max(obs_y) and max(forecasted_y) >= max(obs_y):
            popt_opt = (popt)
            pass

        else:
            
            r2_opt = 0
            popt_opt = 0
            for i in [3]:
                    
                o_y = np.array(obs_y)
                ct = 0
                
                while max(forecasted_y) > i * max(obs_y):
                    ct += 1
                    if ct > 10:
                        break
                    
                    try:
                        popt, pcov = curve_fit(logistic3, 
                                               obs_x, 
                                               o_y, 
                                               sigma= 1 - 1/o_y,
                                               absolute_sigma=True,
                                               method='lm', 
                                               maxfev=40000)
                        
                        pred_y = logistic3(obs_x, *popt)
                        
                        # extend x values by number of ForecastDays
                        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
                        # get corresponding forecasted y values, i.e., extend the predictions
                        forecasted_y = logistic3(forecasted_x, *popt)
                        
                        if max(forecasted_y) > i * max(obs_y):
                            o_y[-1] = o_y[-1] - ((o_y[-1] - o_y[-2]) * 0.5)
                            #o_y[-2:] = o_y[-2:]**0.999
                        
                        else:
                            r2 = obs_pred_rsquare(obs_y, pred_y)
                            if r2 > r2_opt:
                                r2_opt = float(r2)
                                popt_opt = popt
                            
                                
                    except:
                        continue
        
        if r2_opt > 0:
            pred_y = logistic3(obs_x, *popt_opt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = logistic3(forecasted_x, *popt_opt)
                        
                
    except:
        
        try:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            obs_x = np.array(obs_x)
            obs_y = np.array(obs_y)
            forecasted_y = [np.inf]
            forecasted_x = [0]
            pred_y = [np.inf]
            r2_opt = 1
            popt_opt = 1
            try:
                popt, pcov = curve_fit(logistic2, 
                                       obs_x, 
                                       obs_y, 
                                       sigma= 1 - 1/obs_y,
                                       absolute_sigma=True,
                                       method='lm', 
                                       maxfev=40000)
                        
                pred_y = logistic2(obs_x, *popt)
                        
                # extend x values by number of ForecastDays
                forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
                # get corresponding forecasted y values, i.e., extend the predictions
                forecasted_y = logistic2(forecasted_x, *popt)
            
            except:
                pass
            
            if max(forecasted_y) <= 2 * max(obs_y) and max(forecasted_y) >= max(obs_y):
                popt_opt = (popt)
                pass
    
            else:
                
                r2_opt = 0
                popt_opt = 0
                for i in [2]:
                        
                    o_y = np.array(obs_y)
                    ct = 0
                    
                    while max(forecasted_y) > i * max(obs_y):
                        ct += 1
                        if ct > 10:
                            break
                        
                        try:
                            popt, pcov = curve_fit(logistic2, 
                                                   obs_x, 
                                                   o_y, 
                                                   sigma= 1 - 1/o_y,
                                                   absolute_sigma=True,
                                                   method='lm', 
                                                   maxfev=40000)
                            
                            pred_y = logistic2(obs_x, *popt)
                            
                            # extend x values by number of ForecastDays
                            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
                            # get corresponding forecasted y values, i.e., extend the predictions
                            forecasted_y = logistic2(forecasted_x, *popt)
                            
                            if max(forecasted_y) > i * max(obs_y):
                                o_y[-1] = o_y[-1] - ((o_y[-1] - o_y[-2]) * 0.5)
                                #o_y[-2:] = o_y[-2:]**0.999
                            
                            else:
                                r2 = obs_pred_rsquare(obs_y, pred_y)
                                if r2 > r2_opt:
                                    r2_opt = float(r2)
                                    popt_opt = popt
                                
                                    
                        except:
                            continue
            
            if r2_opt > 0:
                pred_y = logistic2(obs_x, *popt_opt)
                forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
                forecasted_y = logistic2(forecasted_x, *popt_opt)
                            
            
                
        except:
            
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            obs_x = np.array(obs_x)
            obs_y = np.array(obs_y)
                
            o_y = np.array(obs_y)
            popt, pcov = curve_fit(logistic1, 
                                               obs_x, 
                                               o_y, 
                                               #sigma= 1 - 1/o_y,
                                               #absolute_sigma=True,
                                               method='lm', 
                                               maxfev=40000)
                        
            pred_y = logistic1(obs_x, *popt)
                        
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = logistic1(forecasted_x, *popt)
    
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
    
    if model == 'Phase Wave':
        forecasted_y, forecasted_x, pred_y, params = get_phase_wave(obs_x, obs_y, ForecastDays)
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
