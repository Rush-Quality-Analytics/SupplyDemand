import pandas as pd
import datetime
import sys 
import numpy as np
from scipy import stats
import requests

pd.set_option('display.max_columns', None)


exclude = ['Recovered', 'Grand Princess', 'Diamond Princess', 'American Samoa', 
           'US', 'American Samoa', 'Northern Mariana Islands', 'Puerto Rico', 
           'Guam', 'Virgin Islands']

df0 = pd.read_csv('data/COVID-TESTING-DF.txt', sep='\t')
df0 = df0.loc[~df0['Province_State'].isin(exclude)]
states = df0['Province_State'].tolist()
JH_dates = df0['date'].tolist()

states_df = pd.read_csv('data/StatePops.csv', sep=',')
states_df = states_df.loc[~states_df['Province/State'].isin(exclude)]


AA_df = pd.read_csv('data/African_American.csv', sep=',')
AA_df = AA_df.loc[~AA_df['State'].isin(exclude)]

Poverty_df = pd.read_csv('data/Poverty.csv', sep=',')
Poverty_df = Poverty_df.loc[~Poverty_df['State'].isin(exclude)]


exclude_abbv = ['AS', 'VI', 'MP', 'GU', 'PR']
#Atlantic_df = pd.read_csv('https://raw.githubusercontent.com/COVID19Tracking/covid-tracking-data/master/data/states_daily_4pm_et.csv', sep=',')
Atlantic_df = requests.get('https://api.covidtracking.com/v1/states/daily.csv')

f = open('data/Atlantic_df.csv', "w")
f.write(Atlantic_df.text)
f.close()

Atlantic_df = pd.read_csv('data/Atlantic_df.csv', sep=',')

Atlantic_df.drop(['hash', 'dateChecked', 'fips', 'posNeg'], axis=1, inplace=True)
Atlantic_df = Atlantic_df.loc[~Atlantic_df['state'].isin(exclude_abbv)]
state_abvs = Atlantic_df['state'].tolist()



dates = Atlantic_df['date'].tolist()
dates_reformat = []
for d1 in dates:
    d2 = datetime.datetime.strptime(str(d1), '%Y%m%d')
    d2 = d2.strftime("%m/%d/%y")
    dates_reformat.append(d2)
    
Atlantic_df['formatted_dates'] = dates_reformat
Atlantic_df['UniqueRow'] = Atlantic_df['formatted_dates'] + '-' + Atlantic_df['state']


pop_sizes = []
colors = []
AA_pop_tot = []
AA_pop_per = []
per_poor = []
unique_rows = []
delta_testing_rate = []
for i, state in enumerate(states):
    
    df_sub = df0[df0['Province_State'] == state]
    testing_rate = df_sub['Testing_Rate'].tolist()
    
    x = list(range(len(testing_rate)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, testing_rate)
    delta_testing_rate.append(slope)
    
    
    state_abv = states_df[states_df['Province/State'] == state]['Abbreviation'].iloc[0]
    u_row = JH_dates[i] + '-' + state_abv
    unique_rows.append(u_row)
    
    pop_size = states_df[states_df['Province/State'] == state].PopSize.iloc[0]
    pop_sizes.append(pop_size)
    
    
    black_pop = AA_df[AA_df['State'] == state]['BlackTotal'].iloc[0]
    AA_pop_tot.append(black_pop)
    
    black_pop_per = AA_df[AA_df['State'] == state]['BlackPerc'].iloc[0]
    AA_pop_per.append(black_pop_per)
    
    poor = Poverty_df[Poverty_df['State'] == state]['2017_2018_avg'].iloc[0]
    per_poor.append(poor)
    
    if state == 'Illinois':
        colors.append('#FECB52')
    else:
        colors.append('#636EFA')
        
    
df0['PopSize'] = pop_sizes
df0['color'] = colors
df0['BlackTotal'] = AA_pop_tot
df0['%Black'] = AA_pop_per
df0['%Poor'] = per_poor
df0['UniqueRow'] = unique_rows
df0['DeltaTestingRate'] = delta_testing_rate


main_df = pd.merge(df0, Atlantic_df, on='UniqueRow')
main_df.tail(10)
main_df.columns = main_df.columns.str.replace('date_x', 'date')

main_df['sqrt_PopSize'] = np.sqrt(main_df['PopSize'].tolist()).tolist()
main_df['Positives per capita'] = main_df.positive/main_df.PopSize
main_df['Negatives per capita'] = main_df.negative/main_df.PopSize
main_df['Percent positive'] = np.round(100 * main_df.positive/main_df.totalTestResults, 2)
main_df['Tests per capita'] = main_df.People_Tested/main_df.PopSize

try:
    main_df.drop(['Unnamed: 0'], axis=1, inplace=True)
except:
    pass


dates = main_df['date'].tolist()

df_today = main_df[main_df['date'] == dates[-1]].copy()  #'08/27/20'
#print(df_today.shape)
#print(df_today['date'])

df_today['log_PopSize'] = np.log10(df_today['PopSize'])
df_today['log_People_Tested'] = np.log10(df_today['People_Tested'])
df_today['log_Confirmed'] = np.log10(df_today['Confirmed'])
df_today['log_Deaths'] = np.log10(df_today['Deaths'])
df_today['log_negative'] = np.log10(df_today['negative'])
df_today['log_positive'] = np.log10(df_today['positive'])
df_today['log_hospitalizedCurrently'] = np.log10(df_today['hospitalizedCurrently'])
df_today['log_inIcuCurrently'] = np.log10(df_today['inIcuCurrently'])
df_today['log_onVentilatorCurrently'] = np.log10(df_today['onVentilatorCurrently'])


print(list(df_today))
hrate = df_today['Hospitalization_Rate'].tolist()
dates = df_today['date'].tolist()
for i, val in enumerate(dates):
    print(val, '  ', hrate[i])    
#sys.exit()

#df_today.to_excel("data/Testing_DataFrame.xlsx")
main_df.to_pickle('data/Testing_Dataframe.pkl')
df_today.to_pickle('data/Testing_Dataframe_Most_Recent_Day.pkl')

