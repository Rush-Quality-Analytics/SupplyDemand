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
    
    
    JH_DATA_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-'
    
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
    
    
    df_confirmed['population size'] = 0
    df_recovered['population size'] = 0
    df_deaths['population size'] = 0
    
    
    df = pd.concat([df_confirmed, df_deaths, df_recovered])
    col1 = df.pop('type')
    col2 = df.pop('population size')
    df.insert(0, 'type', col1)
    df.insert(1, 'population size', col2)
    
    
    df.to_csv('COVID-CASES-DF.txt', sep='\t')

    
    
dataframe()


