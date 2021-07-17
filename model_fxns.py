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
    
    '''
    Determines the proportion of variability in a data set accounted for by a model
    In other words, this determines the proportion of variation explained by the 1:1 line
    in an observed-predicted plot.
    
    Used in various peer-reviewed publications:
        1. Locey, K.J. and White, E.P., 2013. How species richness and total abundance 
        constrain the distribution of abundance. Ecology letters, 16(9), pp.1177-1185.
        2. Xiao, X., McGlinn, D.J. and White, E.P., 2015. A strong test of the maximum 
        entropy theory of ecology. The American Naturalist, 185(3), pp.E70-E80.
        3. Baldridge, E., Harris, D.J., Xiao, X. and White, E.P., 2016. An extensive 
        comparison of species-abundance distribution models. PeerJ, 4, p.e2823.

    '''
    
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)


def gaussian1(x, n, s, m):  
    # For non-cumulative Gaussian:
    #    return n**2 * (1/(s*((2*pi)**0.5))) * np.exp(-0.5 * ((x - m)/s)**2)
    return (n * 0.5 * (1 + sc.special.erf((x - m)/(s*2**0.5))))
    
def gaussian2(x, n, s, m, s1, m1):
    return (n * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5)))))
    
def gaussian3(x, n, s, m, s1, m1, s2, m2):  
    return (n * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5))) + (1 + sc.special.erf((x - m2)/(s2*2**0.5)))))
  
def gaussian4(x, n, s, m, s1, m1, s2, m2, s3, m3):  
    return (n * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5))) + (1 + sc.special.erf((x - m2)/(s2*2**0.5)))) + (1 + sc.special.erf((x - m3)/(s3*2**0.5))))
  
    
def phase_wave1(x,  a, b, c, d, f, g):
    return  (a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b)))
    
def phase_wave2(x,  a, b, c, d, f, g,   a1, b1, c1, d1, g1, f1):
    return  (a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))   +   a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1)))
    
def phase_wave3(x, a, b, c, d, f, g,   a1, b1, c1, d1, g1, f1,   a2, b2, c2, d2, g2, f2):
    return  (a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b)) + a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1)) + a2 / (d2 + np.exp(-c2 * (x + g2*np.sin(f2*x)) + b2)))
        
def phase_wave4(x, a, b, c, d, f, g,   a1, b1, c1, d1, g1, f1,   a2, b2, c2, d2, g2, f2,  a3, b3, c3, d3, g3, f3):
    return  (a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b)) + a1 / (d1 + np.exp(-c1 * (x + g1*np.sin(f1*x)) + b1)) + a2 / (d2 + np.exp(-c2 * (x + g2*np.sin(f2*x)) + b2))) + a3 / (d3 + np.exp(-c3 * (x + g3*np.sin(f3*x)) + b3))


    
def logistic1(x, a, b, c, d):
    return  ((a / (d + np.exp(-c * x + b))))
    
def logistic2(x, a, b, c, d,  a1, b1, c1, d1):
    return  ((a / (d + np.exp(-c * x + b)))   +   (a1 / (d1 + np.exp(-c1 * x + b1))))

def logistic3(x, e, a, b, c, d,  a1, b1, c1, d1,  a2, b2, c2, d2):
    return  ((a / (d + np.exp(-c * x + b)))   +   (a1 / (d1 + np.exp(-c1 * x + b1)))   +   (a2 / (d2 + np.exp(-c2 * x + b2)))) 

def logistic4(x, e, a, b, c, d,  a1, b1, c1, d1,  a2, b2, c2, d2,  a3, b3, c3, d3):
    return  ((a / (d + np.exp(-c * x + b)))   +   (a1 / (d1 + np.exp(-c1 * x + b1)))   +   (a2 / (d2 + np.exp(-c2 * x + b2)))) + (a3 / (d3 + np.exp(-c3 * x + b3))) 





def get_WAF(obs_x, obs_y, ForecastDays):
    
    '''
    WAF: Weighted average forecasting
    
    WAF is a novel time-series analytical (TSA) approach. WAF is model-free and non-parametric.
    WAF is not intended to replace or out-perform other TSA approaches.
    
    WAF is intended to provide no-fail forecasts in relatively little time while being 
    easy to automate, i.e., requirements for web apps with hard time-out limits.
    
    Most TSA approaches, in particular, models in the ARMA/ARIMA family can require the user
    to tune parameters in order to obtain forecastrs. Without tuning those models can 
    fail to reach convergence and fail to return a forecast. Likewise, the run-time of such
    approaches can be lengthy and unpredictable.
    
    Concept: WAF operates by finding a series of future consecutive proportional changes based on
    the history of consecutive proportional changes. 
    
    In this function, WAF operates by:
        
        1. Deconstructing a series of cumulative values, such as daily cumulative numbers 
           of COVID-19 cases, into a series of new daily values. 
        2. Removes sampling error (statistical noise) using scipy's savgol_filter
           function within its signal processing library to smooth the data according to 
           user-defined parameters of window_size and the desired order for the polynomial
           smoothing function (poly_order). This step is technically not necessary but, 
           if not used, the forecasted trend may contain substantial noise, suggesting a 
           greater-than-actual degree of daily precision in forecasted values.
        3. Generating a series of values representing the history of daily proportional 
           change in values.
        4. Generating a combined series of predicted values (corresponding to observed days) 
           and forecasted values (corresponding to days not yet observed). For each daily 
           value, WAF obtains the weighted average daily proportional change. Weights are 
           determined by the similarity between the current proportional change and each 
           historical proportional change. In this way, if the proportional change between yesterday
           and today was 0.1, then ...
    
    
    WAF effectively has 3 parameters:
        1. window_size: used in the savgol_filter smoothing function
        2. poly_order: used in the savgol_filter smoothing function
        3. r: an exponential weight adjustment parameter, the higher this value the 
            exponentially less weight that smaller values (i.e., smaller similarities) have
            on the weighted average
    
    '''
    
    r, window_size, poly_order = 6, 61, 4
    
    def most_likely(y0, n1, n2, r = 0):
        c = 10**-r
        wts = (c/(c + np.abs(y0 - n1))) ** r # weights associated with each value
                # in the list of values for historical proportional change (n1)
                
        # The 1/(1+x) term transforms greater differences into greater similarities
        # with values ranging beween 0 and 1.
                
        # The '** 10**-r' terms adds a double exponential response to the weights, giving
        # exponentially greater weight to greater similarities but keeping the min
        # and max weight values between 0 and 1.
                
        # The user can define whatever other weighting function seems best to them.
                
        exp_y = np.average(n2, weights=wts)
        return exp_y


    def smooth(x, poly_order, window_size):
        x = savgol_filter(x, window_size, poly_order)
        return x
    
    
    def get_results(obs_y, ForecastDays):
        # obs_y: list of cumulative values
        # ForecastDays: no. of values beyond the observed values that you want forecasted
        
        # replace 0's with 1's, otherwise 0's may find their way into denominators
        obs_y = [1 if x == 0 else x for x in obs_y]
        l = len(obs_y)
        
        ##### GENERATE AND SMOOTH A LIST OF DAILY NEW VALUES ######
        
        # n0 is a list holding the number of new cases reported for each day in the time series.
        # We assume the number of new cases reported on the first day is equal to the first 
        # value in obs_y, since nothing was reported before the first day.
        
        n0 = [obs_y[0]]
        for i, val in enumerate(obs_y):
            if i > 0:
                n0.append(obs_y[i] - obs_y[i-1])
        
        # Smooth the list of daily new values
        n0 = smooth(n0, poly_order, window_size)
        
        ##### GENERATE TWO TIME-STAGGERED LISTS ######
        
        # n1 is a list of daily proportional changes
        # n2 is a time-staggered copy of n1. So, if the first element of n1 is the number of new cases
        # reported on the first day, then the first element of n2 is the number of new cases
        # reported on the second day.
        
        n1 = [0]
        for i, val in enumerate(n0):
            if i > 0:
                
                #if n0[i] == 0:
                #    n0[i] = 1
                #if n0[i-1] == 0:
                #    n0[i-1] = 1
                    
                n1.append((n0[i] - n0[i-1])/n0[i-1])

        n2 = np.array(n1[1:])
        # The last element of n1 has to be removed because there is no complement to it in n2.
        n1 = np.array(n1[:-1])
        
        # Initiate the list of predicted values with the first value in n1. 
        pred_y = [obs_y[1]]
        # Loop through the list of daily new values
        for i, y1 in enumerate(n0):
            
            if i == 0: continue # cannot get a prediction for the first day because
                                # there is not a value for the day before it.
        
            pc = (n0[i] - n0[i-1])/(n0[i-1]) # proportional change between 'yesterday' 
                                             # and 'today', with respect to the loop
                                             
            pc = most_likely(pc, n1, n2, r)
            y2 = y1 + pc * y1 # expected daily value for 'tomorrow', with respect to the loop 
                
            pred_y.append(y2)
        
        ##### GET FORECASTED ######
        
        forecasted_y = [pred_y[-2], pred_y[-1]]
        for i in range(ForecastDays - 1):
            
            if i == 0: continue # cannot get a prediction for the first day because
                                # there isn't a value for the day before it.
            
            if forecasted_y[i-1] == 0:
                # zero values will become the denominator (below) and produce an error
                forecasted_y[i-1] = 1

            pc = (forecasted_y[i] - forecasted_y[i-1])/(forecasted_y[i-1])
            pc = most_likely(pc, n1, n2, r)
            y2 = forecasted_y[i] + pc * forecasted_y[i]
                
            if y2 < 0:
                y2 = 0
            forecasted_y.append(y2)
              
        forecasted_y = pred_y + forecasted_y[1:]
        
        pred_y = smooth(pred_y, poly_order, window_size)
        forecasted_y = smooth(forecasted_y, poly_order, window_size)
        
        return pred_y, forecasted_y
    
    
    #### GENERATE PREDICTED Y-VALUES AND FORECASTED Y-VALUES
    # predicted y-values are simply those corresponding to observed days
    # forecasted y-values are those corresponding to days not yet observed
    pred_y, forecasted_y = get_results(obs_y, ForecastDays)
    
    
    ### CONVERTE PREDICTED AND FORECASTED VALUES TO CUMULATIVE VALUES
    pred_y = [sum(pred_y[0:x:1]) for x in range(len(pred_y)+1)][:-1]
    forecasted_y = [sum(forecasted_y[0:x:1]) for x in range(len(forecasted_y)+1)][:-1]
    
    
    ### CHECK FOR UNREASONABLE GROWTH AND REDUCE THE LAST FEW OBSERVED Y-VALUES IF NECESSARY
    ct = 0 
    i = 2
    while max(forecasted_y) > 2 * max(obs_y) and ct < 100:
        
        # If the max forecasted value is greater than twice the max observed value,
        # then we assume the TSA has predicted an unrealistic rate of growth.
        # If that happens, then we reduce the last i values by 10% of the difference between 
        # those values and the values that came before them.
        # We then rerun the TSA, which should produce less unrealistic growth.
        # We perform this procedure a maximum of 100 times.
        
        # This step is not necessary and can be modified or excluded. But, the result 
        # may produce unrealistic growth that exceeds the size of the focal population.
        
        # This heuristic is only due to the need for the COVID calculator to not produce 
        # unrealistic 'doomsday' scenarios, and has worked well for Rush University Medical Center.
        
        obs_y[-i:] = obs_y[-i:] - ((obs_y[-i:] - obs_y[-i-1]) * 0.1)
        
        pred_y, forecasted_y = get_results(obs_y, ForecastDays)
        pred_y = [sum(pred_y[0:x:1]) for x in range(len(pred_y)+1)][:-1]
        forecasted_y = [sum(forecasted_y[0:x:1]) for x in range(len(forecasted_y)+1)][:-1]
        
        ct += 1
    
    
    forecasted_x = list(range(len(forecasted_y)))
    params = [r, window_size, poly_order] # if the user prefers, they can add the TSA parameters (r, window_size, poly_order)
    
    return forecasted_y, forecasted_x, pred_y, params




def opt_fit(obs_y, obs_x, forecasted_y, ForecastDays, model):
    
    # A FUNCTION TO IMPROVE SCIPY'S ABILITY TO FIND A FITTED CURVE
    # This is done because scipy's numerical optimizer may fail to converge and return
    # a prediction that, e.g., rapidly accelerates to infinity, negative infinity, etc.
    # This is often driven by large changes in the few most recent observed values.
    
    # This function is not called by the time-series analysis (WAF)
    
    # Process:
    # 1. substract a fraction (cf) of the difference between the most recent i values and the 
         # value that preceeded each.
    # 2. Rerun the model for a maximum of 10 times (to not exceed time-out limits), updating the 
         # resulting 'optimal' parameters (popt_opt) when the r-square improves. Importantly,
         # the r-square is based on a comparison of predicted values to the original y-values.
         
    # 3. Rerun the model a final time using the so-called optimal parameters, i.e., those
         # producing the highest r-square
        
    # Note: This step is not necessary and can be modified or excluded. But, the result 
    # may produce unrealistic growth that exceeds the size of the focal population.
        
    # Note: This heuristic is only due to the need for the COVID calculator to avoid 
    # unrealistic 'doomsday' scenarios or nonsensical predictions for hospital systems
    # for all states and counties
    
        
    cf = 0.5
    i = 4
    
    r2_opt = 0
    ct = 0
    
    popt_opt = 0
    o_y = np.array(obs_y)
    o_y[-i:] = o_y[-i:] - ((o_y[-i:] - o_y[-i-1]) * cf)

        
    while max(forecasted_y) > i * max(obs_y):
        ct += 1
        if ct > 10:
            break
        
        
        if model == 'phase_wave4':
            popt, pcov = curve_fit(phase_wave4, obs_x, o_y, 
                                   #sigma= 1 - 1/o_y,
                                   #absolute_sigma=True, 
                                   method='lm', 
                                   maxfev=20000)
            
            pred_y = phase_wave4(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = phase_wave3(forecasted_x, *popt)
            
        elif model == 'phase_wave3':
            popt, pcov = curve_fit(phase_wave3, obs_x, o_y, 
                                   #sigma= 1 - 1/o_y,
                                   #absolute_sigma=True, 
                                   method='lm', 
                                   maxfev=20000)
            
            pred_y = phase_wave3(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = phase_wave3(forecasted_x, *popt)
                
        elif model == 'phase_wave2':
            popt, pcov = curve_fit(phase_wave2, obs_x, o_y, 
                                   #sigma= 1 - 1/o_y,
                                   #absolute_sigma=True, 
                                   method='lm', 
                                   maxfev=20000)
            
            pred_y = phase_wave2(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = phase_wave2(forecasted_x, *popt)
        
        elif model == 'logistic4':
            popt, pcov = curve_fit(logistic4, obs_x, o_y, 
                                   #sigma= 1 - 1/o_y,
                                   #absolute_sigma=True, 
                                   method='lm', 
                                   maxfev=20000)
            
            pred_y = logistic4(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = logistic3(forecasted_x, *popt)
            
        elif model == 'logistic3':
            popt, pcov = curve_fit(logistic3, obs_x, o_y, 
                                   #sigma= 1 - 1/o_y,
                                   #absolute_sigma=True, 
                                   method='lm', 
                                   maxfev=20000)
            
            pred_y = logistic3(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = logistic3(forecasted_x, *popt)
                
        elif model == 'logistic2':
            popt, pcov = curve_fit(logistic2, obs_x, o_y, 
                                   #sigma= 1 - 1/o_y,
                                   #absolute_sigma=True, 
                                   method='lm', 
                                   maxfev=20000)
            
            pred_y = logistic2(obs_x, *popt)
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            forecasted_y = logistic2(forecasted_x, *popt)
            
        r2 = obs_pred_rsquare(obs_y, pred_y)
        if r2 > r2_opt:
            r2_opt = float(r2)
            popt_opt = popt
            
        o_y[-i:] = o_y[-i:] - ((o_y[-i:] - o_y[-i-1]) * cf)
    
    
    if model == 'phase_wave4':
        pred_y = phase_wave4(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = phase_wave3(forecasted_x, *popt_opt)
        
    elif model == 'phase_wave3':
        pred_y = phase_wave3(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = phase_wave3(forecasted_x, *popt_opt)
                
    elif model == 'phase_wave2':
        pred_y = phase_wave2(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = phase_wave2(forecasted_x, *popt_opt)
        
    elif model == 'logistic4':
        pred_y = logistic4(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = logistic3(forecasted_x, *popt_opt)
        
    elif model == 'logistic3':
        pred_y = logistic3(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = logistic3(forecasted_x, *popt_opt)
                
    elif model == 'logistic2':
        pred_y = logistic2(obs_x, *popt_opt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = logistic2(forecasted_x, *popt_opt)
            
    return pred_y, forecasted_x, forecasted_y




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
        popt, pcov = curve_fit(phase_wave4, obs_x, obs_y,
                               method='lm', maxfev=20000)
        # get predicted y values
        pred_y = phase_wave4(obs_x, *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = phase_wave4(forecasted_x, *popt)
        
        r2 = obs_pred_rsquare(obs_y, pred_y)
        if r2 in [np.inf, -np.inf, np.nan]:
            g = 1 + []
            
        elif max(forecasted_y) > 3*max(obs_y):
            model = 'phase_wave4'
            pred_y, forecasted_x, forecasted_y = opt_fit(obs_y, obs_x, forecasted_y, ForecastDays, model)
            
    except:
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
        popt, pcov = curve_fit(logistic4, obs_x, obs_y,
                               method='lm', maxfev=20000)
        # get predicted y values
        pred_y = logistic4(obs_x, *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = logistic4(forecasted_x, *popt)
        
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
                               method='lm', maxfev=20000)
        
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
                               method='lm', maxfev=20000)
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
                               method='lm', maxfev=20000)
        
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
                               method='lm', maxfev=20000)
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
    
    
    fy = np.array(fy)
    return fy, forecasted_x, pred_y, params






def fit_curve(condition):
    
    obs_x1, obs_y1, obs_x, obs_y, model, ForecastDays, day, iterations = condition
    # A function to fit various models to observed COVID-19 cases data according to:
    # obs_x: observed x values
    # obs_y: observed y values
    # model: the model to fit
    # ForecastDays: number of days ahead to extend predictions
    # N: population size of interest

    # In the following, the observed vs. predicted r-squares are based only on the last 
    # 30-days of observed data
    
    # use the number of y observations as the number of x observations
    obs_x1 = list(range(len(obs_y1)))
    obs_x = list(range(len(obs_y)))
    # convert y and x observations to numpy arrays
    obs_x1 = np.array(obs_x1)
    obs_y1 = np.array(obs_y1)
    obs_x = np.array(obs_x)
    obs_y = np.array(obs_y)
    
    Miny = np.min(obs_y)
    if model != 'Time series analysis':
        obs_y = obs_y - Miny
    
    # Get the forecasted values, predicted values, and observed vs predicted r-square
    # value for the chosen model
    
    if model == 'Phase Wave':
        forecasted_y, forecasted_x, pred_y, params = get_phase_wave(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
    
    elif model == 'Time series analysis':
        forecasted_y, forecasted_x, pred_y, params = get_WAF(obs_x1, obs_y1, ForecastDays)
        
        pred_y = pred_y[len(obs_y1)-len(obs_y):]
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
        
        forecasted_y = forecasted_y[len(obs_y1)-len(obs_y):]
        forecasted_x = forecasted_x[len(obs_x1)-len(obs_x):]
        
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
        
    elif model == '2 phase sine-logistic': 
        # This model is no longer used in Rush University Medical Center's COVID app.
        
        # This model was included in the associated publication:
            # Locey, K.J., Webb, T.A., Khan, J., Antony, A.K. and Hota, B., 2020. 
            # An interactive tool to forecast US hospital needs in the coronavirus 2019 pandemic. medRxiv.
        
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
        
        params = params1.extend(params2)
        
    
    elif model == '2 phase logistic':
        
        # This model is no longer used in Rush University Medical Center's COVID app.
        
        # This model was included in the associated publication:
            # Locey, K.J., Webb, T.A., Khan, J., Antony, A.K. and Hota, B., 2020. 
            # An interactive tool to forecast US hospital needs in the coronavirus 2019 pandemic. medRxiv.
        
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
       
        
    #print(len(pred_y), len(obs_y), len(obs_x), ':', len(forecasted_y), len(forecasted_x))
    
    if model != 'Time series analysis':
        pred_y = np.array(pred_y) + Miny
        pred_y = pred_y.tolist()
        forecasted_y = np.array(forecasted_y) + Miny
        forecasted_y = forecasted_y.tolist()
    
    del obs_y
    
    return [obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params]
