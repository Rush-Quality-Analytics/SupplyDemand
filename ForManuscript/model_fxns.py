import pandas as pd # data frame library

import datetime # library for date-time functionality
import numpy as np # numerical python
import scipy as sc
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
    
    
    def gaussian(x, n, s, m):  
        #return n**2 * (1/(s*((2*pi)**0.5))) * np.exp(-0.5 * ((x - m)/s)**2)
        return n**2 * 0.5 * (1 + sc.special.erf((x - m)/(s*2**0.5)))


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
        popt, pcov = curve_fit(gaussian, obs_x, obs_y, method='lm', maxfev=2000)
        # get predicted y values
        pred_y = gaussian(obs_x, *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = gaussian(forecasted_x, *popt)
        
        
    except:
        # if the logistic model totally fails to fit
        # then use the polynomial model
        forecasted_y, forecasted_x, pred_y, params = get_polynomial(obs_x, obs_y, ForecastDays, 3)
    
    # return the forecasted x-y values and predicted y values
    params = []
    
    
    return forecasted_y, forecasted_x, pred_y, params




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





def get_logistic(obs_x, obs_y, ForecastDays):
    
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
                               method='lm', maxfev=2000,
                               )
        
        
        
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
        forecasted_y, forecasted_x, pred_y, params = get_gaussian(obs_x, obs_y, ForecastDays)
        return forecasted_y, forecasted_x, pred_y, params
    
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
    return forecasted_y, forecasted_x, pred_y, params
        


def get_polynomial(obs_x, obs_y, ForecastDays, degree=2):
    
    def cubic(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d
    
    def quadratic(x, a, b, c):
        return a*x**2 + b*x + c
    
    
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
        if degree == 3:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(cubic, obs_x, obs_y, method='lm', maxfev=2000)
            # get predicted y values
            pred_y = cubic(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = cubic(forecasted_x, *popt)
            
            
        if degree == 2:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(quadratic, obs_x, obs_y, method='lm', maxfev=2000)
            # get predicted y values
            pred_y = quadratic(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = quadratic(forecasted_x, *popt)
                
    except:
        # if the polynomial model fails, the lack of a substitute here
        # will throw an error
        pass
        
    params = []
    # return the forecasted x-y values and predicted y values
    return forecasted_y, forecasted_x, pred_y, params




def fit_curve(obs_x, obs_y, model, ForecastDays, N, ArrivalDate, day, iterations):
    
    # A function to fit various models to observed COVID-19 cases data according to:
    # obs_x: observed x values
    # obs_y: observed y values
    # model: the model to fit
    # ForecastDays: number of days ahead to extend predictions
    # N: population size of interest
    # T0: likely data of first infection (used by SEIR-SD model)
    # incubation_period: disease-specific epidemilogical parameter
        # average number of days until an exposed person becomes
        # begins to exhibit symptoms of infection
    # infectious_period: disease-specific epidemilogical parameter
        # average number of days an infected person is infected
    # rho: disease-specific epidemilogical parameter
        # aka basic reproductive number
        # average number of secondary infections produced by a typical case 
        # of an infection in a population where everyone is susceptible
    # socdist: population-specific social-distancing parameter
    
    
    # use the number of y observations as the number of x observations
    obs_x = list(range(len(obs_y)))
    # convert y and x observations to numpy arrays
    obs_x = np.array(obs_x)
    obs_y = np.array(obs_y)
    
    
    # Get the forecasted values, predicted values, and observed vs predicted r-square
    # value for the chosen model
    
    if model == '2 phase sine-logistic':
        
        max_r2 = 0
        b_pt = 10
        for i in range(0, len(obs_x)):
            
            try:
        
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
                
                obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
                if obs_pred_r2 > max_r2:
                    max_r2 = obs_pred_r2
                    b_pt = i
                    
            except:
                continue
        
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
            
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
            
        
        params = params1.extend(params2)
        
    elif model == 'Sine-logistic':
        forecasted_y, forecasted_x, pred_y, params = get_sine_logistic(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
    
    
    elif model == '2 phase logistic':
        
        max_r2 = 0
        b_pt = 10
        for i in range(0, len(obs_x)):
        
            try:
                obs_x1 = obs_x[0:i]
                obs_y1 = obs_y[0:i]
            
                obs_x2 = obs_x[i:]
                obs_y2 = obs_y[i:]
                
                
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
                
                obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
                if obs_pred_r2 > max_r2:
                    max_r2 = obs_pred_r2
                    b_pt = i
                    
            except:
                continue
        
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
            
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
            
        
        params = params1.extend(params2)
        
    
    
    
    elif model == 'Logistic':
        forecasted_y, forecasted_x, pred_y, params = get_logistic(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
        
    elif model == 'Gaussian':
        forecasted_y, forecasted_x, pred_y, params = get_gaussian(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
    
    elif model == 'Exponential':
        forecasted_y, forecasted_x, pred_y, params = get_exponential(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
    
    elif model == '3rd degree polynomial':
        forecasted_y, forecasted_x, pred_y, params = get_polynomial(obs_x, obs_y, ForecastDays, 3)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
    
    elif model == 'Quadratic':
        forecasted_y, forecasted_x, pred_y, params = get_polynomial(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
        
    elif model == 'SEIR-SD':
        forecasted_y, forecasted_x, pred_y, params = get_seir_sd(obs_x, obs_y, ForecastDays, N, iterations, day)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
        
    # return the r-square and observed, predicted, and forecasted values
    return obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params











def get_seir_sd(obs_x, obs_y, ForecastDays, N, iterations, day):
    
    def correct_beta(sd, beta, i):
        # A function to adjust the contact rate (beta) by the percent infected
        
        beta = beta/(sd*i + 1)
        
        return beta



    def test_effect(i):
        # A logistic function with an output ranging between 0 and 1 
        # 'i' is the time step
        # This function corrects the apparent number of infected 
        # according to an assumption that testing for COVID-19 was
        # minimal in the first few weeks of infection
        
        return 1/(1+np.exp(-0.1*i+5.8))
    
    
    def seir(obs_x, alpha, beta, gamma, d1, sd, s, im, fdays=0):
        
        sN = int(N)
        today = pd.to_datetime('today', format='%Y/%m/%d')
        ArrivalDate = today - datetime.timedelta(days = d1 + len(obs_x))
        t_max = (today - ArrivalDate).days
        
        t_max += fdays
        t = list(range(int(t_max))) 
        
        # unpack initial values
        S = [1 - 1/sN] # fraction susceptible
        E = [1/sN] # fraction exposed
        I = [0] # fraction infected
        cI = [0]
        R = [0] # fraction recovered
        
        # declare a list that will hold testing-corrected
        # number of infections
        Ir = [0]
        
        # loop through time steps from date of first likely infection
        
        # to end of forecast window
        
        for i in t[1:]:
            
            #print('hi', t_max, alpha, beta, gamma)
            
            # fraction infected is the last element in the I list
    
            # adjust the contact rate (beta) by the % infected
            beta = correct_beta(sd, beta, I[-1])
            
            # No. susceptible at time t = S - beta*S*I
            next_S = S[-1] - beta * S[-1] * I[-1]
            
            # No. exposed at time t = S - beta*S*I - alpha*E
            next_E = E[-1] + beta * S[-1] * I[-1] - alpha * E[-1] 
            
            # No. infected at time t = I + alpha*E - gamma*I
            next_I = I[-1] + alpha * E[-1] - gamma * I[-1] 
            
            next_cI = cI[-1] + alpha * E[-1]
            
            # No. recovered at time t = R - gamma*I
            next_R = R[-1] + gamma*I[-1] #+ gamma*Q[-1]
            
            
            tS = next_S * sN
            tE = next_E * sN
            tI = next_I * sN
            tR = next_R * sN
            tcI = next_cI * sN
            
            sN = sN + im*sN
            tE = tE + im*sN
            
            next_S = tS/sN
            next_E = tE/sN
            next_I = tI/sN
            next_R = tR/sN
            next_cI = tcI/sN
            
            
            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
            cI.append(next_cI)
            
            # get the testing lag
            test_lag = test_effect(i)
            # get apparent infected by multiplying I by the testing lag
            Ir.append(next_cI * test_lag)
        
        # multiply array of apparent percent infected by N
        I = np.array(Ir) * sN
        # convert I to a list
        I = I.tolist()
        
        
        # number of observed days + size of forecast window
        num = len(obs_x)+fdays
        
        # trim extra days, i.e., between T0 and day of first observation
        I = I[-num:]
        I = np.array(I)
        
        return I



    forecasted_y, forecasted_x, pred_y = [], [], []
    pred_y_o = []
    alpha_o, beta_o, gamma_o, d1_o, sd_o, s_o, im_o = 1,1,1,1,1,1,1
    
    incubation_period = 1/alpha_o
    infectious_period = 1/gamma_o
    rho_o = beta_o/gamma_o
    
    
    rho = float(rho_o)
    alpha = float(alpha_o)
    beta = float(beta_o)
    gamma = float(gamma_o)
    d1 = float(d1_o)
    sd = float(sd_o)
    s = float(s_o)
    im = float(im_o)
    
    r2_o = 0


    obs_x = obs_x.tolist()
    
    iterations = 200000
    for i in range(iterations):
        
        pred_y = seir(obs_x, alpha, beta, gamma, d1, sd, s, im)   
        obs_pred_r2 = obs_pred_rsquare(obs_y[0:], pred_y[0:])
        
        if i == 0 or obs_pred_r2 >= r2_o:
            
            alpha_o = float(alpha)
            beta_o = float(beta)
            gamma_o = float(gamma)
            d1_o = float(d1)
            sd_o = float(sd)
            s_o = float(s)
            r2_o = float(obs_pred_r2)
            pred_y_o = list(pred_y)
            im_o = float(im)
            
        incubation_period = np.random.uniform(4, 6)
        infectious_period = np.random.uniform(4, 14)
        rho = np.random.uniform(1, 6)
        sd = np.random.uniform(1, 100)
        d1 = np.random.randint(1, 60)
        s = np.random.uniform(0, 1)
        im = 10**np.random.uniform(-7, -3)
        
        alpha = 1/incubation_period 
        gamma = 1/infectious_period
        beta = rho*gamma
    
    
    
    forecasted_y = seir(obs_x, alpha_o, beta_o, gamma_o, d1_o, sd_o, s_o, im_o, ForecastDays-1)
    params = [alpha_o, beta_o, gamma_o, d1_o, sd_o, s_o, im_o]
    
    # return the forecasted x-y values and predicted y values
    return forecasted_y, forecasted_x, pred_y_o, params