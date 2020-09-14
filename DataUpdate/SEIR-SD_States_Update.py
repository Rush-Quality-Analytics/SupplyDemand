import pandas as pd # data frame library
import datetime # library for date-time functionality
import numpy as np # numerical python
import model_fxns as fxns




model_fits_df = pd.read_csv('data/SEIR-SD_States_Update.txt', sep='\t')
model_fits_df.drop(model_fits_df.columns[1], axis=1, inplace=True)

df_sub = model_fits_df[model_fits_df['focal_loc'] == 'Illinois']
ref_date = model_fits_df['reference_date'].iloc[-1]

model = 'SEIR-SD'

ap_df = pd.read_csv('data/COVID-CASES-DF.txt', sep='\t') 
ap_df = ap_df[ap_df['Country/Region'] == 'US']
ap_df = ap_df[ap_df['Province/State'] != 'US']
ap_df = ap_df[ap_df['Province/State'] != 'American Samoa']
ap_df = ap_df[ap_df['Province/State'] != 'Northern Mariana Islands']
ap_df.drop(columns=['Unnamed: 0'], inplace=True)

StatePops = pd.read_csv('data/StatePops.csv')
StatePops = StatePops[StatePops['Province/State'] != 'American Samoa']
StatePops = StatePops[StatePops['Province/State'] != 'Northern Mariana Islands']

states = list(set(StatePops['Province/State']))
states.sort()


df_sub = ap_df[ap_df['Province/State'] == 'Illinois']
yi = list(df_sub)
DATES_O = yi[4:]



if ref_date == DATES_O[-1]:
    print('latest day in data:', DATES_O[-1])
    print("up to date")


else:
    
    for focal_loc in states:
        
        ForecastDays = 60
        PopSize = StatePops[StatePops['Province/State'] == focal_loc]['PopSize'].tolist()
        PopSize = PopSize[0]
            
        ArrivalDate = StatePops[StatePops['Province/State'] == focal_loc]['Date_of_first_reported_infection'].tolist()
        ArrivalDate = ArrivalDate[0]
            
        ForecastDays = int(ForecastDays+1)
            
            
        # filter main dataframe to include only the chosen location
        df_sub = ap_df[ap_df['Province/State'] == focal_loc]
            
        # get column labels, will filter below to extract dates
        yi = list(df_sub)
        
            
        y = df_sub.iloc[0,4:].values
        DATES_O = yi[4:]
        
        
        
        obs_y = []
        i = 0
        while y[i] == 0: i+=1
        obs_y = y[i:]
        DATES_O = DATES_O[i:]
        
        num_days = [0]
        
        for i, j in enumerate(num_days):
            
            
            if j == 0:
                # get dates for today's predictions/forecast
                DATES = DATES_O[0:]
                obs_y_trunc = obs_y[0:]
                #if len(obs_y_trunc) == 0 or max(obs_y_trunc) == 1:
                #    continue
            
            # declare x as a list of integers from 0 to len(y)
            x = list(range(len(obs_y_trunc)))
             
            iterations = 200000
            SEIR_fit = 0
            
            
            obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params = fxns.fit_curve(x, obs_y_trunc, 
                                    model, ForecastDays, PopSize, ArrivalDate, j, iterations, SEIR_fit)
            
             
            print(focal_loc, obs_pred_r2, len(params))
            
            if obs_pred_r2 < 0:
                obs_pred_r2 = 0.0
    
            # convert any y-values (observed, predicted, or forecasted)
            # that are less than 0 (nonsensical values) to 0.
            obs_y_trunc[obs_y_trunc < 0] = 0
            pred_y = np.array(pred_y)
            pred_y[pred_y < 0] = 0
    
            forecasted_y = np.array(forecasted_y)
            forecasted_y[forecasted_y < 0] = 0
            
            # number of from ArrivalDate to end of forecast window
            #numdays = len(forecasted_x)
            latest_date = pd.to_datetime(DATES[-1])
            first_date = pd.to_datetime(DATES[0])
    
            # get the date of the last day in the forecast window
            future_date = latest_date + datetime.timedelta(days = ForecastDays-1)
                
            # get all dates from ArrivalDate to the last day in the forecast window
            fdates = pd.date_range(start=first_date, end=future_date)
            fdates = fdates.strftime('%m/%d')
            
            model_fits_df.loc[len(model_fits_df)] = [obs_y_trunc, pred_y, DATES, 
                                                     forecasted_y, fdates, 
                                                     model, focal_loc, 
                                                     obs_pred_r2, params, 
                                                     DATES[-1]]
    
    
    model_fits_df.to_csv('data/SEIR-SD_States_Update.txt', sep='\t')
        