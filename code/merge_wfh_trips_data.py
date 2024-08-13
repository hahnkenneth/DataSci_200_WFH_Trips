#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:17:28 2024

@author: kennethhahn
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

# %% Read Data
trips_df = pd.read_csv('./Trips_by_Distance.csv')
wfh_df = pd.read_excel('./remote_work_in_job_ads_signup_data.xlsx',sheet_name = 'us_state_by_month')
wfh_df_national = pd.read_excel('./remote_work_in_job_ads_signup_data.xlsx',sheet_name = 'country_by_month')


#%% Commonly Used Functions

def column_value_counts(df):
    '''This function goes through each column of DataFrame and prints out
    the unique values of the DataFrame'''
    for col in df.columns:
        print(df[col].value_counts())
        print('-'*50)
def column_nan_counts(df):
    '''This function goes thorugh each column of a DataFrame and prints out 
    the number of blank values in each column'''
    for col in df.columns:
        num_blanks = df[col].isna().sum()
        len_col = len(df[col])
        print(f'{col} has {num_blanks} blank values/'
              f'which is {100*(num_blanks/len_col):.2f}% blanks')
    print('-'*50)
def column_lower(df):
    '''Makes all column names of a DataFrame lowercase and also removes spaces from it'''
    columns = df.columns.str.lower().str.replace(' ','_')
    df.columns = columns

#%% Understand and clean trips_df

# Standardize column names
column_lower(trips_df)

# make date column into DateTime object
trips_df['date'] = pd.to_datetime(trips_df['date'])

# Only look at data pertaining to states
state_trips_df = trips_df[trips_df['level']=='State']

# Get national data as well
national_trips_df = trips_df[trips_df['level']=='National']

# Learn the types of values in each column and num nan values
column_value_counts(state_trips_df)
column_nan_counts(state_trips_df)
# only columns that has blanks is County FIPS and County Name which makes sense

# print number of state
num_states = len(state_trips_df['state_postal_code'].value_counts())
print(f'There are {num_states} states in this dataset')
print('-'*50)

# Looks like Washington DC is the 51st state included in this dataset
states_list_trips = state_trips_df['state_postal_code'].value_counts().index

# I want to check if all the different trips columns summed together ~equals the number_of_trips column
number_of_trips_columns = [col for col in state_trips_df.columns if col.startswith('number_of_trips_')]

# sum the trips columns and check to see if the different trip columns adds up to number_of_trips column within +/- 1 tolerance
state_trips_df['sum_trips'] = state_trips_df[number_of_trips_columns].sum(axis=1)
state_trips_df['approx_equal'] = np.isclose(state_trips_df['number_of_trips'], state_trips_df['sum_trips'], rtol=1)

# this adds up all the rows and sees what percentage of the rows does the number of trips of each column == the number_of_trips
check_sum_equality = 100*state_trips_df['approx_equal'].sum()/len(state_trips_df)
# looks like all of them add up correctly, which is good

#%% Understand and clean wfh_df

# Standardize column names
column_lower(wfh_df)

# making a new month_year datetime column
wfh_df['month_year'] = wfh_df['month'] + ' ' + wfh_df['year'].astype('str')
wfh_df['month_year'] = pd.to_datetime(wfh_df['month_year'], format='%b %Y')

# Learn the types of values in each column and num nan values
column_value_counts(wfh_df)
column_nan_counts(wfh_df)
# no nan values, which is good 

# print number of state
num_states = len(wfh_df['state'].value_counts())
print(f'There are {num_states} states in this dataset')
print('-'*50)

# This data set also includes Washington DC so the data is standardized between both DataFrames
states_list_wfh = wfh_df['state_code'].value_counts().index

# Standardize column names
column_lower(wfh_df)

# Only get US data
wfh_df_national = wfh_df_national[wfh_df_national['country'] == 'US']

# making a new month_year datetime column
wfh_df_national['month_year'] = wfh_df_national['month'] + ' ' + wfh_df_national['year'].astype('str')
wfh_df_national['month_year'] = pd.to_datetime(wfh_df_national['month_year'], format='%b %Y')

# Learn the types of values in each column and num nan values
column_value_counts(wfh_df_national)
column_nan_counts(wfh_df_national)
# no nan values, which is good 

# %% Get total population

state_trips_df['total_population'] = state_trips_df['population_staying_at_home'] + state_trips_df['population_not_staying_at_home']
national_trips_df['total_population'] = national_trips_df['population_staying_at_home'] + national_trips_df['population_not_staying_at_home']

total_population_state = state_trips_df[['date','state_postal_code','total_population']].rename(columns={'state_postal_code':'state_code'})
total_population_nation = national_trips_df[['date','total_population']]

total_population_state['month'] = total_population_state['date'].dt.month
total_population_state['year'] = total_population_state['date'].dt.year
total_population_state['month_year'] = total_population_state['month'].astype('str') + '-' + total_population_state['year'].astype('str')
total_population_state['month_year'] = pd.to_datetime(total_population_state['month_year'], format='%m-%Y')

total_population_nation['month'] = total_population_nation['date'].dt.month
total_population_nation['year'] = total_population_nation['date'].dt.year
total_population_nation['month_year'] = total_population_nation['month'].astype('str') + '-' + total_population_nation['year'].astype('str')
total_population_nation['month_year'] = pd.to_datetime(total_population_nation['month_year'], format='%m-%Y')

monthly_population_state = total_population_state.groupby(['month_year', 'state_code']).mean().reset_index()
monthly_population_nation = total_population_nation.groupby(['month_year']).mean().reset_index()

monthly_population_state = monthly_population_state.drop(columns = ['date','month','year'])
monthly_population_nation = monthly_population_nation.drop(columns = ['date','month','year'])

#%% Merge the two DataFrames together

nostr_state_trips = state_trips_df

# make a month_year datetime column to merge with wfh_df
nostr_state_trips['month'] = nostr_state_trips['date'].dt.month
nostr_state_trips['year'] = nostr_state_trips['date'].dt.year
nostr_state_trips['month_year'] = nostr_state_trips['month'].astype('str') + '-' + nostr_state_trips['year'].astype('str')
nostr_state_trips['month_year'] = pd.to_datetime(nostr_state_trips['month_year'], format='%m-%Y')

# drop all columns with strings in them to conduct groupby
nostr_state_trips = nostr_state_trips.drop(columns=['level','county_fips','county_name','row_id','date'])

# rename state_postal_code to state_code to make merging easier with wfh_df
nostr_state_trips = nostr_state_trips.rename(columns={'state_postal_code':'state_code'})

# groupby state_trips_df by month_year and state, taking the sum of each month and state
monthly_trips_df = nostr_state_trips.groupby(['month_year', 'state_code']).sum().reset_index()

# merge the two DataFrames. Only merge left since there is less month_year data from the trips
# DataFrame compared to to the wfh_df
trips_wfh_df = monthly_trips_df.merge(wfh_df,on=['month_year','state_code'],how = 'left')

# drop redundant columns
columns_to_drop = ['month_x','month_y','year_x','year_y','year_month','measurement','week','state_fips','approx_equal','sum_trips']
trips_wfh_df = trips_wfh_df.drop(columns = columns_to_drop)

#trips_wfh_df.to_csv('./Kenneth_merged_dataset.csv')

# %% Merge the two DataFrames toether

nostr_national_trips = national_trips_df

# make a month_year datetime column to merge with wfh_df
nostr_national_trips['month'] = nostr_national_trips['date'].dt.month
nostr_national_trips['year'] = nostr_national_trips['date'].dt.year
nostr_national_trips['month_year'] = nostr_national_trips['month'].astype('str') + '-' + nostr_national_trips['year'].astype('str')
nostr_national_trips['month_year'] = pd.to_datetime(nostr_national_trips['month_year'], format='%m-%Y')

# drop all columns with strings in them to conduct groupby
nostr_national_trips = nostr_national_trips.drop(columns=['level','state_fips','state_postal_code','county_fips','county_name','row_id','date'])

# groupby state_trips_df by month_year and state, taking the sum of each month and state
monthly_national_trips_df = nostr_national_trips.groupby(['month_year']).sum().reset_index()

# merge the two DataFrames. Only merge left since there is less month_year data from the trips
# DataFrame compared to to the wfh_df
national_trips_wfh_df = monthly_national_trips_df.merge(wfh_df_national,on=['month_year'],how = 'left')

# drop redundant columns
columns_to_drop = ['month_x','month_y','year_x','year_y','year_month','week']
national_trips_wfh_df = national_trips_wfh_df.drop(columns = columns_to_drop)

# trips_wfh_df.to_csv('./Kenneth_merged_dataset.csv')

#%% merge with  total_population tables
trips_wfh_df = trips_wfh_df.drop(columns=['total_population'])
national_trips_wfh_df = national_trips_wfh_df.drop(columns=['total_population'])

new_trips_wfh_df = trips_wfh_df.merge(monthly_population_state,on=['month_year','state_code'])
new_national_trips_wfh_df = national_trips_wfh_df.merge(monthly_population_nation,on=['month_year'])

new_trips_wfh_df.to_excel('./Merged Trips and WFH by State with Correct Total Population.xlsx')
new_national_trips_wfh_df.to_excel('./Merged Trips and WFH by Nation with Correct Total Population.xlsx')