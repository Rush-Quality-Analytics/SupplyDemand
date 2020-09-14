import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import datetime
import numpy as np

dates = []
def dataframe():
    JH_DATA_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'
    df_main = pd.DataFrame(columns = ['Province/State', 'Country/Region', 'date',
                                'Lat' , 'Long', 'Confirmed', 'Deaths', 'Recovered'])
    
    
    today = pd.Timestamp('today')
    today = '{:%m-%d-%Y}'.format(today)
    
    
    dates = pd.date_range(start='3-10-2020', end=today, freq='d')
    dates = pd.to_datetime(dates, format='%m-%d-%Y')
    dates = dates.strftime('%m-%d-%Y').tolist()
    
    for date in dates:
        fname = JH_DATA_URL + date + '.csv'
        
        try:
            df = pd.read_csv(fname)
        except:
            continue
        
        d1 = datetime.datetime.strptime(date, "%m-%d-%Y")
        d2 = datetime.datetime.strptime('03-22-2020', "%m-%d-%Y")
        
        if d1 < d2:
            try:
                df = df[df['Country/Region'] == 'US']
                df['date'] = date
                df = df.filter(['Province/State', 'Country/Region', 'date',
                                'Lat' , 'Long', 'Confirmed', 'Deaths', 'Recovered'])
                
            except:
                pass
            
        else:
            try:
                df = df[df['Country_Region'] == 'US']
                df['date'] = date
                df = df.filter(['Province_State', 'Country_Region', 'date', 
                                'Lat' , 'Long_', 'Confirmed', 'Deaths', 'Recovered'])
                
                df.columns = ['Province/State', 'Country/Region', 'date',
                                'Lat' , 'Long', 'Confirmed', 'Deaths', 'Recovered']
            except:
                pass
            
        df_main = pd.concat([df_main, df])
    
    df_main['date'] = pd.to_datetime(df_main['date'])
    df_main['date'] = df_main['date'].dt.strftime('%m/%d/%Y')

    try:
        df_main['date'] = df_main['date'].map(lambda x: x.rstrip('0'))
        df_main['date'] = df_main['date'].map(lambda x: x.rstrip('2'))
        df_main['date'] = df_main['date'].map(lambda x: x.lstrip('0'))
    except:
        pass

    df_sums = df_main.groupby(['Province/State','date'])['Confirmed'].sum().reset_index()
    
    return df_sums, df_main, dates
    

df_sums, df_main, dates = dataframe()


#print(list(df_sums))
#print(df_sums.shape)
#print(len(dates))


col_names = ['Province/State', 'Country/Region', 'Lat', 'Long']

today = pd.Timestamp('today')
today = '{:%m/%d/%Y}'.format(today)
dates = pd.date_range(start='3/10/2020', end=today, freq='d')
dates = pd.to_datetime(dates, format='%m/%d/%Y')
dates = dates.strftime('%m/%d/%Y')

dates = dates.map(lambda x: x.rstrip('0'))
dates = dates.map(lambda x: x.rstrip('2'))
dates = dates.map(lambda x: x.lstrip('0'))



col_names.extend(dates)

provstates = list(set(df_sums['Province/State']))

lofl = []
for ps in provstates:
    st_df = df_main[df_main['Province/State'].str.contains(ps)]
    lat = st_df['Lat'].iloc[-1]
    long = st_df['Long'].iloc[-1]
    
    ps_ls = [ps, 'US', lat, long]
    cases = df_sums[df_sums['Province/State'].str.contains(ps)]
    st_cases = cases['Confirmed'].tolist()
    st_dates = cases['date'].tolist()
    
    c = []
    for date in dates:
        try:
            ii = st_dates.index(date)
            c.append(st_cases[ii])
        except:
            c.append(0)

    ps_ls.extend(c)
    lofl.append(ps_ls)

    
df = pd.DataFrame.from_records(lofl, columns=col_names)

c = df.shape[1] - 1
sum_col = np.array(df.iloc[:, c:].T)[0]

if sum(sum_col) == 0:
    df = df.drop(df.columns[c], axis=1)


df.to_csv('data/COVID-CASES-DF.txt', sep='\t')