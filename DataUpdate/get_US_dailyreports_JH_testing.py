import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import datetime
import numpy as np

dates = []
def dataframe():
    JH_DATA_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'
    df_main = pd.DataFrame(columns = ['Province_State', 'date',
                                'Confirmed', 'Deaths', 'Recovered',
                                'Incident_Rate', 'People_Tested', 'People_Hospitalized',
                                'Mortality_Rate', 'Testing_Rate', 'Hospitalization_Rate'])
    
    
    today = pd.Timestamp('today')
    today = '{:%m-%d-%Y}'.format(today)
    
    
    dates = pd.date_range(start='4-12-2020', end=today, freq='d')
    dates = pd.to_datetime(dates, format='%m-%d-%Y')
    dates = dates.strftime('%m-%d-%Y').tolist()
    
    for date in dates:
        fname = JH_DATA_URL + date + '.csv'
        
        try:
            df = pd.read_csv(fname)
        except:
            continue
        
        d1 = datetime.datetime.strptime(date, "%m-%d-%Y")
        d2 = datetime.datetime.strptime('04-12-2020', "%m-%d-%Y")
        
        if d1 < d2:
            try:
                df = df[df['Country/Region'] == 'US']
                df['date'] = date
                df = df.filter(['Province_State', 'date',
                                'Confirmed', 'Deaths', 'Recovered',
                                'Incident_Rate', 'People_Tested', 'People_Hospitalized',
                                'Mortality_Rate', 'Testing_Rate', 'Hospitalization_Rate'])
                
            except:
                pass
            
        else:
            try:
                df = df[df['Country_Region'] == 'US']
                df['date'] = date
                df = df.filter(['Province_State', 'date',
                                'Confirmed', 'Deaths', 'Recovered',
                                'Incident_Rate', 'People_Tested', 'People_Hospitalized',
                                'Mortality_Rate', 'Testing_Rate', 'Hospitalization_Rate'])
                
                df.columns = ['Province_State', 'date',
                                'Confirmed', 'Deaths', 'Recovered',
                                'Incident_Rate', 'People_Tested', 'People_Hospitalized',
                                'Mortality_Rate', 'Testing_Rate', 'Hospitalization_Rate']
            except:
                pass
            
        df_main = pd.concat([df_main, df])
    
    df_main['date'] = pd.to_datetime(df_main['date'])
    df_main['date'] = df_main['date'].dt.strftime('%m/%d/%y')

    #try:
    #    df_main['date'] = df_main['date'].map(lambda x: x.rstrip('0'))
    #    df_main['date'] = df_main['date'].map(lambda x: x.rstrip('2'))
    #    df_main['date'] = df_main['date'].map(lambda x: x.lstrip('0'))
    #except:
    #    pass

    return df_main, dates
    

df_main, dates = dataframe()
df_main.to_csv('data/COVID-TESTING-DF.txt', sep='\t')