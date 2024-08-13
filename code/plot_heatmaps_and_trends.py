#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python file takes from the merge_wfh_trips_data.py module to get the merged data
with the trips dataset and the wfh dataset.

It will use this data to perform different aggregations to make heat maps, scatter plots,
bar plots, and box plots..
"""

from merge_wfh_trips_data import new_trips_wfh_df, new_national_trips_wfh_df 

#%% import all used libraries.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import os
import wget
import matplotlib.colors as mcolors

# %% group by state and also by covid time period. Also clean up the columns to shorter names
trips_wfh_df = new_trips_wfh_df
national_trips_wfh_df = new_national_trips_wfh_df

columns_of_interest = ['state_code', 'number_of_trips_<1', 'number_of_trips_1-3',
                       'number_of_trips_3-5', 'number_of_trips_5-10', 'number_of_trips_10-25', 
                       'number_of_trips_25-50', 'number_of_trips_50-100', 'number_of_trips_100-250', 
                       'number_of_trips_250-500', 'number_of_trips_>=500','number_of_trips','total_population']

columns_of_interest2 = ['month_year','state_code', 'number_of_trips_<1', 'number_of_trips_1-3',
                       'number_of_trips_3-5', 'number_of_trips_5-10', 'number_of_trips_10-25', 
                       'number_of_trips_25-50', 'number_of_trips_50-100', 'number_of_trips_100-250', 
                       'number_of_trips_250-500', 'number_of_trips_>=500','number_of_trips','total_population']


subset_df = trips_wfh_df[columns_of_interest]
subset_df2 = trips_wfh_df[columns_of_interest2]

def sum_mean(column):
    '''This function is used in the agg group by to take an average over the 
    total population column while summing all other columns'''
    if column.name == 'total_population':
        return column.mean()
    else:
        return column.sum()

def group_by_timeframe(df,timeframe):
    '''This function will help with expediting the groupby in order to create separate dataframes
    from before, during, and after covid, grouped by state. It will also calculate the per capita number
    of trips per state'''
    pre_covid = pd.to_datetime('2020-04-01')
    post_covid = pd.to_datetime('2021-02-01')
    
    if timeframe  == 'pre':
        grouped_df = df[df['month_year'] < pre_covid]
    elif timeframe == 'during':
        grouped_df = df[(df['month_year'] >= pre_covid) & (df['month_year'] <= post_covid)]
    elif timeframe == 'post':
        grouped_df = df[df['month_year'] > post_covid]
    
    grouped_df = grouped_df.drop(columns='month_year')
    grouped_df_new = grouped_df.groupby(['state_code']).agg(sum_mean)
    for col in grouped_df_new.columns[:-1]:
        grouped_df_new[col] = grouped_df_new[col]/(grouped_df_new['total_population'])
    
    new_column_names = [col.split('_')[3] if col.startswith('number_of_trips_') else col for col in grouped_df_new.columns]
    grouped_df_new.columns = new_column_names
    
    return grouped_df_new

    
grouped_trips = subset_df.groupby(['state_code']).agg(sum_mean)

state_trips_monthly = trips_wfh_df

precovid_grouped = group_by_timeframe(subset_df2,'pre')
duringcovid_grouped = group_by_timeframe(subset_df2,'during')
postcovid_grouped = group_by_timeframe(subset_df2,'post')

for col in grouped_trips.columns[:-1]:
    if col not in ('month_year','state_code','total_population'):
        grouped_trips[col] = grouped_trips[col]/grouped_trips['total_population']
        state_trips_monthly[col] = state_trips_monthly[col]/state_trips_monthly['total_population']
    

new_column_names = [col.split('_')[3] if col.startswith('number_of_trips_') else col for col in grouped_trips.columns]
grouped_trips.columns = new_column_names

new_column_names1 = [col.split('_')[3] if col.startswith('number_of_trips_') else col for col in national_trips_wfh_df.columns]
national_trips_wfh_df.columns = new_column_names1

# %% Heat map of short trips (trips less than 10 miles)

# Plot the heatmap
plt.figure(figsize=(13, 15))
sns.heatmap(grouped_trips.iloc[:,:-2], cmap='viridis', annot=True)
plt.title('Heatmap of Number of Trips by State and Trip Distance (Per Capita)',fontsize=15,fontweight='bold')
plt.xlabel('Trip Distance (Miles Travelled)',fontsize=15,fontweight='bold')
plt.ylabel('State Code',fontsize=15,fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=12)

ax = plt.gca()
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)

#%% find miles travelled by multiplying the population size in each bucked by the midpoint of the bucket range. Then sum the number of miles into one main column.
grouped_trips_miles = subset_df.groupby(['state_code']).agg(sum_mean)
grouped_trips_miles.columns = new_column_names
for col in grouped_trips_miles.columns[:-1]:
    if col == '<1':
        grouped_trips_miles[col] *= 0.5
    elif col == '1-3':
        grouped_trips_miles[col] *= 2
    elif col == '3-5':
        grouped_trips_miles[col] *= 4
    elif col == '5-10':
        grouped_trips_miles[col] *= 7.5
    elif col == '10-25':
        grouped_trips_miles[col] *= 17.5
    elif col == '25-50':
        grouped_trips_miles[col] *= 37.5
    elif col == '50-100':
        grouped_trips_miles[col] *= 75
    elif col == '100-250':
        grouped_trips_miles[col] *= 175
    elif col == '250-500':
        grouped_trips_miles[col] *= 375
    elif col == '>=500':
        grouped_trips_miles[col] *= 500

grouped_trips_miles['total_miles'] = grouped_trips_miles.iloc[:,:10].sum(axis = 1)
grouped_trips_miles['total_miles_per_capita'] = grouped_trips_miles['total_miles']/grouped_trips_miles['total_population']
grouped_trips_miles['total_trips_per_capita'] = grouped_trips_miles['number_of_trips']/grouped_trips_miles['total_population']

#%% plot miles travelled by state and trips taken by state
grouped_trips_miles = grouped_trips_miles.sort_values('total_miles_per_capita',ascending=False)
plt.figure(figsize = (20,6))
plt.bar(grouped_trips_miles.index,grouped_trips_miles['total_miles_per_capita'])
plt.xlabel('State',fontsize=15)
plt.ylabel('Miles Travelled (per capita)',fontsize=15)
plt.xlim(left=-0.5,right = 50.5)
plt.title('Miles Travelled by State',fontsize=20)

grouped_trips_miles = grouped_trips_miles.sort_values('total_trips_per_capita',ascending=False)
plt.figure(figsize = (20,6))
plt.bar(grouped_trips_miles.index,grouped_trips_miles['total_trips_per_capita'],color = 'orange')
plt.xlabel('State',fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=14)
plt.ylabel('Trips Taken (per capita)',fontsize=15)
plt.title('Trips Taken by State Comparison Plot',fontsize=20)
plt.xlim(left=-0.5,right = 50.5)

# %% Plot total_population by state
grouped_trips_miles = grouped_trips_miles.sort_values('total_population',ascending=False)
plt.figure(figsize = (20,6))
plt.bar(grouped_trips_miles.index,grouped_trips_miles['total_population'],color = 'green')
plt.xlabel('State',fontsize=15)
plt.ylabel('Total Population by State',fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=14)
plt.title('Population By State Comparison Plot',fontsize=20)
plt.xlim(left=-0.5,right = 50.5)
plt.gca().yaxis.major.formatter._useMathText = True  # Enable math text mode
plt.gca().yaxis.offsetText.set_fontsize(15)  # Set fontsize for the scientific notation

#%% Do the same for national data

national_trips_wfh_df = national_trips_wfh_df.set_index('month_year')
national_trips_wfh_df = national_trips_wfh_df.drop(columns = ['country'])

#%% Same as above. Calculate the number of miles travelled by state
national_trips_miles = national_trips_wfh_df
for col in national_trips_miles.columns:
    if col == '<1':
        national_trips_miles[col] *= 0.5
    elif col == '1-3':
        national_trips_miles[col] *= 2
    elif col == '3-5':
        national_trips_miles[col] *= 4
    elif col == '5-10':
        national_trips_miles[col] *= 7.5
    elif col == '10-25':
        national_trips_miles[col] *= 17.5
    elif col == '25-50':
        national_trips_miles[col] *= 37.5
    elif col == '50-100':
        national_trips_miles[col] *= 75
    elif col == '100-250':
        national_trips_miles[col] *= 175
    elif col == '250-500':
        national_trips_miles[col] *= 375
    elif col == '>=500':
        national_trips_miles[col] *= 500

national_trips_miles['total_miles'] = national_trips_miles.iloc[:,3:13].sum(axis = 1)
national_trips_miles['total_miles_per_capita'] = national_trips_miles['total_miles']/national_trips_miles['total_population']
national_trips_miles['total_trips_per_capita'] = national_trips_miles['number_of_trips']/national_trips_miles['total_population']

#%% Plot the time trend of WFH by month and total trips per capita by month on the same plot.

national_trips_miles = national_trips_miles[national_trips_miles.index < pd.to_datetime('2023-10-01')]

fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(national_trips_miles.index,national_trips_miles['total_trips_per_capita'],marker='o',color='blue',label='Total Trips Per Capita')
ax1.set_ylabel('Number of Trips per capita',color='blue')
ax1.axvline(pd.to_datetime('2020-04-01'),linestyle = '--',color='black',label = 'Start of COVID-19 Outbreak')
ax1.axvline(pd.to_datetime('2021-02-01'),linestyle = ':',color='black',label = 'End of COVID-19 Outbreak')
ax2 = ax1.twinx()
ax2.plot(national_trips_miles.index,national_trips_miles['percent'],marker = 's',color='red',label='Percent of Remote Work Jobs')
ax2.set_ylabel('Percent of Remote Work Jobs',color='red')
fig.legend(loc = 'center left',bbox_to_anchor=(0.641,0.2))
ax1.set_title('Time Series of Number of Trips Per Capita and Percent of Remote Work Jobs')

#%% Define the COVID-19 period and plot the scatterplot to see relationships between WFH and number of trips per capita
#Also create a strip plot to see how the number of trips changes before and after COVID

conditions = [(national_trips_miles.index < pd.to_datetime('2020-04-01')),
              (national_trips_miles.index >= pd.to_datetime('2020-04-01')) & (national_trips_miles.index <= pd.to_datetime('2021-02-01')),
              (national_trips_miles.index > pd.to_datetime('2021-02-01'))]

conditions_state = [(state_trips_monthly['month_year'] < pd.to_datetime('2020-04-01')),
              (state_trips_monthly['month_year'] >= pd.to_datetime('2020-04-01')) & (state_trips_monthly['month_year'] <= pd.to_datetime('2021-02-01')),
              (state_trips_monthly['month_year'] > pd.to_datetime('2021-02-01'))]

values = ['Before COVID-19','During COVID-19','After COVID-19']

national_trips_miles['timeframe'] = np.select(conditions,values)
state_trips_monthly['timeframe'] = np.select(conditions_state,values)


plt.figure(figsize=(8,6))
sns.scatterplot(data=national_trips_miles,x='percent',y='total_trips_per_capita',hue='timeframe')
plt.ylabel('Number of Trips (per capita)',fontsize=12)
plt.xlabel('Percent of Remote Job Postings',fontsize=12)
plt.title('Total Trips vs Remote Job Postings',fontsize=15)
plt.ylim([0,145])

plt.figure(figsize=(8,6))
sns.stripplot(national_trips_miles,x='timeframe',y='total_trips_per_capita',hue='timeframe',dodge=True)
sns.boxplot(data=national_trips_miles,x='timeframe',y='total_trips_per_capita',hue='timeframe',dodge=True,boxprops=dict(alpha=0.5), showfliers=False)
plt.ylabel('Number of Trips (per capita)',fontsize=12)
plt.xlabel('Timeframe',fontsize=12)
plt.title('Total Trips for each Timeframe',fontsize=15)
plt.ylim([0,145])

# %% Create the same scatter plot as above but do it by some select states (top 3 in the bar plots from above)

states = state_trips_monthly['state_code'].unique()
states_of_interest = ['DC','CA','TX','VT','MA','NY','VA','FL']
def states_wfh_scatter(df,state,ax=None):
    sns.scatterplot(df[df['state_code']==state],x='percent',y='number_of_trips',hue='timeframe',ax=ax,legend=False)
    ax.set_ylim([0,200])
    # plt.ylabel('Number of Trips (per capita)',fontsize=12)
    # plt.xlabel('Percent of Remote Job Postings',fontsize=12)
    # plt.title(f'Total Trips vs Remote Job Postings for {state}',fontsize=15)

fig,ax = plt.subplots(4,2,figsize=(12,12),sharex=True)
ax = ax.flatten()

for i,state in enumerate(states_of_interest):
    current_axis = ax[i]
    states_wfh_scatter(state_trips_monthly,state,ax[i])
    current_axis.set_title(f'Total Trips vs Remote Job Postings for {state}')
    current_axis.set_ylabel('Number of Trips (per capita)')
    current_axis.set_xlabel('Percent of Remote Jobs')

plt.tight_layout()

# Get the handles and labels of the last axis to create a main legend
handles, labels = ax[-1].get_legend_handles_labels()

# Create a main legend using handles and labels from the last axis
legend_labels = ['Before COVID-19','During COVID-19','After COVID-19']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in sns.color_palette()[:3]]
fig.legend(legend_handles, legend_labels, loc='lower center',bbox_to_anchor=(0.5,-0.03), ncol=len(legend_labels), fontsize=12)


# Show the plot with the main legend
plt.show()


# %% geo heat map data processing for COVID-19

grouped_trips['mean'] = grouped_trips.iloc[:,:-2].mean(axis=1)
geo_data = grouped_trips['mean'].reset_index()

precovid_grouped['mean_pre'] = precovid_grouped.iloc[:,:-2].mean(axis=1)
print('mean_pre min max', (precovid_grouped['mean_pre'].min(),precovid_grouped['mean_pre'].max()))
geo_data_pre = precovid_grouped['mean_pre'].reset_index()
duringcovid_grouped['mean_during'] = duringcovid_grouped.iloc[:,:-2].mean(axis=1)
print('mean_during min max', (duringcovid_grouped['mean_during'].min(),duringcovid_grouped['mean_during'].max()))
geo_data_during= duringcovid_grouped['mean_during'].reset_index()
postcovid_grouped['mean_post'] = postcovid_grouped.iloc[:,:-2].mean(axis=1)
print('mean_post min max', (postcovid_grouped['mean_post'].min(),postcovid_grouped['mean_post'].max()))
geo_data_post = postcovid_grouped['mean_post'].reset_index()

# %% Download zip file for shape data of the US with borders for each state (extract the contents of the zip file once opened)
wget.download("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip")

# %% Read data from the zip file and merge with the pre, during, and post covid data from two cells ago
gdf = gpd.read_file(os.getcwd()+'/cb_2018_us_state_500k')
gdf = gdf.merge(geo_data_pre,left_on='STUSPS',right_on='state_code')
gdf = gdf.merge(geo_data_during,left_on='STUSPS',right_on='state_code')
gdf = gdf.merge(geo_data_post,left_on='STUSPS',right_on='state_code')

# %% Map hawaii and Alaska separately to make them look nicer on the US map
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# NOTE: the convention for polygon points is (Long, Lat)....counterintuitive
polygon = Polygon([(-175,50),(-175,72),(-140, 72),(-140,50)])
# polygon = Polygon([(-180,0),(-180,90),(-120,90),(-120,0)])

# polygon=hipolygon
poly_gdf = gpd.GeoDataFrame( geometry=[polygon], crs=world.crs)

polygon = Polygon([(-170,50),(-170,72),(-140, 72),(-140,50)])
# apply1(alaska_gdf,0,36)
alaska_gdf = gdf[gdf.state_code=='AK']
hawaii_gdf = gdf[gdf.state_code=='HI']
hipolygon = Polygon([(-161,0),(-161,90),(-120,90),(-120,0)])
hawaii_gdf.clip(hipolygon).plot(color='lightblue', linewidth=0.8, edgecolor='0.8')
# %% Add the frames for Alaska and Hawaii onto the US plot. Also create a function for defining the heat map scale

# Create a "copy" of gdf for re-projecting
visframe = gdf.to_crs({'init':'epsg:2163'})

# create figure and axes for with Matplotlib for main map
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box from the main map
ax.axis('off')


# create map of all states except AK and HI in the main map axis
visframe[~visframe.state_code.isin(['HI','AK'])].plot(color='lightblue', linewidth=0.8, ax=ax, edgecolor='0.8')


# Add Alaska Axis (x, y, width, height)
akax = fig.add_axes([0.1, 0.17, 0.17, 0.16])   


# Add Hawaii Axis(x, y, width, height)
hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])   

# We'll later map Alaska in "akax" and Hawaii in "hiax"
def makeColorColumn(gdf,variable,vmin,vmax):
    # apply a function to a column to create a new column of assigned colors & return full frame
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlOrBr)
    gdf['value_determined_color'] = gdf[variable].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
    return gdf

# %% This will plot the actual heat map of the US by state for data pre-pandemic
variable = 'mean_pre'

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf['mean_pre'].min(), gdf['mean_pre'].max() #math.ceil(gdf.pct_food_insecure.max())
# Choose the continuous colorscale "YlOrBr" from https://matplotlib.org/stable/tutorials/colors/colormaps.html
colormap = "YlOrBr"


gdf = makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163
visframe = gdf.to_crs({'init':'epsg:2163'})



# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

# set the font for the visualization to Helvetica
hfont = {'fontname':'Helvetica'}

# add a title and annotation
ax.set_title('Average Trips Per Capita Choropleth Before COVID-19', **hfont, fontdict={'fontsize': '30', 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

cbax.set_title('Number of\nTrips Per Capita', **hfont, fontdict={'fontsize': '15', 'fontweight' : '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, \
                 norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
fig.colorbar(sm, cax=cbax)
tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)


# create map
# Note: we're going state by state here because of unusual coloring behavior when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state_code not in ['AK','HI']:
        vf = visframe[visframe.state_code==row.state_code]
        c = gdf[gdf.state_code==row.state_code][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

# add Alaska
akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])   
akax.axis('off')
# polygon to clip western islands
polygon = Polygon([(-170,50),(-170,72),(-140, 72),(-140,50)])
alaska_gdf = gdf[gdf.state_code=='AK']
alaska_gdf.clip(polygon).plot(color=gdf[gdf.state_code=='AK'].value_determined_color, linewidth=0.8,ax=akax, edgecolor='0.8')


# add Hawaii
hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])   
hiax.axis('off')
# polygon to clip western islands
hipolygon = Polygon([(-160,0),(-160,90),(-120,90),(-120,0)])
hawaii_gdf = gdf[gdf.state_code=='HI']
hawaii_gdf.clip(hipolygon).plot(column=variable, color=hawaii_gdf['value_determined_color'], linewidth=0.8,ax=hiax, edgecolor='0.8')

# %% Plots heatmap during the pandemic
variable = 'mean_during'

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf['mean_during'].min(), gdf['mean_during'].max() #math.ceil(gdf.pct_food_insecure.max())
# Choose the continuous colorscale "YlOrBr" from https://matplotlib.org/stable/tutorials/colors/colormaps.html
colormap = "YlOrBr"


gdf = makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163
visframe = gdf.to_crs({'init':'epsg:2163'})



# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

# set the font for the visualization to Helvetica
hfont = {'fontname':'Helvetica'}

# add a title and annotation
ax.set_title('Average Trips Per Capita Choropleth During COVID-19', **hfont, fontdict={'fontsize': '30', 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

cbax.set_title('Number of\nTrips Per Capita', **hfont, fontdict={'fontsize': '15', 'fontweight' : '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, \
                 norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
fig.colorbar(sm, cax=cbax)
tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)


# create map
# Note: we're going state by state here because of unusual coloring behavior when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state_code not in ['AK','HI']:
        vf = visframe[visframe.state_code==row.state_code]
        c = gdf[gdf.state_code==row.state_code][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

# add Alaska
akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])   
akax.axis('off')
# polygon to clip western islands
polygon = Polygon([(-170,50),(-170,72),(-140, 72),(-140,50)])
alaska_gdf = gdf[gdf.state_code=='AK']
alaska_gdf.clip(polygon).plot(color=gdf[gdf.state_code=='AK'].value_determined_color, linewidth=0.8,ax=akax, edgecolor='0.8')


# add Hawaii
hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])   
hiax.axis('off')
# polygon to clip western islands
hipolygon = Polygon([(-160,0),(-160,90),(-120,90),(-120,0)])
hawaii_gdf = gdf[gdf.state_code=='HI']
hawaii_gdf.clip(hipolygon).plot(column=variable, color=hawaii_gdf['value_determined_color'], linewidth=0.8,ax=hiax, edgecolor='0.8')

# %% Plots the heatmap post pandemic
variable = 'mean_post'

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf['mean_post'].min(), gdf['mean_post'].max() #math.ceil(gdf.pct_food_insecure.max())
# Choose the continuous colorscale "YlOrBr" from https://matplotlib.org/stable/tutorials/colors/colormaps.html
colormap = "YlOrBr"


gdf = makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163
visframe = gdf.to_crs({'init':'epsg:2163'})



# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

# set the font for the visualization to Helvetica
hfont = {'fontname':'Helvetica'}

# add a title and annotation
ax.set_title('Average Trips Per Capita Choropleth After COVID-19', **hfont, fontdict={'fontsize': '30', 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

cbax.set_title('Number of\nTrips Per Capita', **hfont, fontdict={'fontsize': '15', 'fontweight' : '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, \
                 norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
fig.colorbar(sm, cax=cbax)
tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)


# create map
# Note: we're going state by state here because of unusual coloring behavior when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state_code not in ['AK','HI']:
        vf = visframe[visframe.state_code==row.state_code]
        c = gdf[gdf.state_code==row.state_code][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

# add Alaska
akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])   
akax.axis('off')
# polygon to clip western islands
polygon = Polygon([(-170,50),(-170,72),(-140, 72),(-140,50)])
alaska_gdf = gdf[gdf.state_code=='AK']
alaska_gdf.clip(polygon).plot(color=gdf[gdf.state_code=='AK'].value_determined_color, linewidth=0.8,ax=akax, edgecolor='0.8')


# add Hawaii
hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])   
hiax.axis('off')
# polygon to clip western islands
hipolygon = Polygon([(-160,0),(-160,90),(-120,90),(-120,0)])
hawaii_gdf = gdf[gdf.state_code=='HI']
hawaii_gdf.clip(hipolygon).plot(column=variable, color=hawaii_gdf['value_determined_color'], linewidth=0.8,ax=hiax, edgecolor='0.8')

