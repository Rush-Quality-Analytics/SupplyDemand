#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
import sys


def dataframe():
    ### Pull online data:
    ### 1. COVID-19: John's Hopkins (https://github.com/CSSEGISandData/COVID-19)
    ### 2. COVID-19: Sito del Dipartimento della Protezione Civile, Italia (https://github.com/pcm-dpc/COVID-19)
    ### 3. Population size: klocey/COVID-dash/master/worldcities.csv
    
    
    JH_DATA_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-'
    Italia_DATA_URL = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv'
    CITIES_URL = 'https://raw.githubusercontent.com/klocey/COVID-dash/master/worldcities.csv'
    
    
    df_confirmed = pd.read_csv(JH_DATA_URL + 'Confirmed.csv', index_col=False)
    ser = pd.Series(df_confirmed.index.tolist()).astype(str)
    df_confirmed['Province/State'].fillna(df_confirmed['Country/Region'] + '_' + ser, inplace=True)
    df_confirmed['type'] = 'Confirmed'
    
    
    df_deaths = pd.read_csv(JH_DATA_URL + 'Deaths.csv', index_col=False)
    ser = pd.Series(df_deaths.index.tolist()).astype(str)
    df_deaths['Province/State'].fillna(df_confirmed['Country/Region'] + '_' + ser, inplace=True)
    df_deaths['type'] = 'Deaths'
            
        
    df_recovered = pd.read_csv(JH_DATA_URL + 'Recovered.csv', index_col=False)
    ser = pd.Series(df_recovered.index.tolist()).astype(str)
    df_recovered['Province/State'].fillna(df_confirmed['Country/Region'] + '_' + ser, inplace=True)
    df_recovered['type'] = 'Recovered'        
    
    
    ProvState = df_confirmed['Province/State']
    #CountryRegion = df_confirmed['Country/Region']
    
    
    WorldCities = pd.read_csv(CITIES_URL, index_col=False)
    places = ProvState.tolist()
    
    
    popsizes = []
    for place in places:
        popsize = 0
        p_ls = ['Taiwan', 'San Diego County', 'Humboldt County', 'Sacramento County']
        s_ls = [23545963, 3095313, 132646, 1418788]
            
        if place in p_ls:
            i = p_ls.index(place)
            popsize = s_ls[i]
            
        else:
            try:
                pattern = re.compile(r'\w\,')
                if pattern.findall(place):
                    place = ','.join(place.split(',')[:-1])    
        
                subdf = WorldCities[WorldCities.isin([place]).any(1)]
                popsize = sum(subdf['population'])
            except:
                pass
        popsizes.append(popsize)
        
        
    df_confirmed['population size'] = list(popsizes)
    df_recovered['population size'] = list(popsizes)
    df_deaths['population size'] = list(popsizes)
    
    
    df = pd.concat([df_confirmed, df_deaths, df_recovered])
    col1 = df.pop('type')
    col2 = df.pop('population size')
    df.insert(0, 'type', col1)
    df.insert(1, 'population size', col2)
    
    
    yi = list(df)
    dates = yi[6:]
    
    #print(dates)
    #sys.exit()
    
    #Italia_DATA_URL = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/531ff2459f941705d85c1c37b972bc66a6bbd5eb/dati-province/dpc-covid19-ita-province.csv'
    df_Italia = pd.read_csv(Italia_DATA_URL, sep='\,', error_bad_lines=False)
    
    #print(list(df_Italia))
    #sys.exit()
    
    dates = df_Italia['data'].tolist()
    
    
    try:
        df_Italia['data'] = df_Italia['data'].map(lambda x: x.rstrip(u"18:0"))
    except:
        pass
    try:
        df_Italia['data'] = df_Italia['data'].map(lambda x: x.rstrip(u"17:0"))
    except:
        pass
    
    df_Italia['data'] = pd.to_datetime(df_Italia['data'])
    df_Italia['data'] = df_Italia['data'].dt.strftime('%m/%d/%Y')
    
    
    try:
        df_Italia['data'] = df_Italia['data'].map(lambda x: x.rstrip('0'))
        df_Italia['data'] = df_Italia['data'].map(lambda x: x.rstrip('2'))
        df_Italia['data'] = df_Italia['data'].map(lambda x: x.lstrip('0'))
    except:
        pass
    
    
    #df_ItaliaT = df_Italia.T
    Italia_df  = pd.DataFrame(columns = list(df))
    Italia_df['type'] = 'Confirmed'
    Italia_df['population size'] = 0
    

    df_temp = df_Italia.filter(items=['denominazione_provincia', 'lat', 'long'])
    df_temp.drop_duplicates(inplace=True)
    
    
    Italia_df['Province/State'] = df_temp['denominazione_provincia']
    Italia_df['Country/Region'] = 'Italia'
    Italia_df['Lat'] = df_temp['lat']
    Italia_df['Long'] = df_temp['long']
    Italia_df = Italia_df.fillna(0)
    
    
    df_Italia = df_Italia.fillna(0)
    for index, row in Italia_df.iterrows():
        p = row['Province/State']
        df_temp = df_Italia[df_Italia['denominazione_provincia'] == p] 
        dates = df_temp['data'].tolist()
        
        cases = df_temp['totale_casi'].tolist()
        
        for i, date in enumerate(dates):
            Italia_df.set_value(index, date, int(cases[i]))
            
            
    Italia_df = Italia_df[df.columns]
    Italia_df['type'] = ['Confirmed']*Italia_df.shape[0]
    
    
    dfMain = pd.concat([Italia_df, df], ignore_index=True)
    #df_ProvState = dfMain.dropna(how='any')
    
    #print(list(df_ProvState))
    
    dfMain.to_csv('COVID-CASES-DF.txt', sep='\t')
    
    return dfMain

    
    
dataframe()


