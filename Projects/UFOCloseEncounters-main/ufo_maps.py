# -*- coding: utf-8 -*-
"""UFO_Maps.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jCH0GyXQhZL5qWE-xOeWOb8lGwbz1aC-

# God Bless Dave Fisher-Hickey!  
Reference: https://www.kaggle.com/daveianhickey/how-to-folium-for-maps-heatmaps-time-data/

# Import Libraries
"""

import pandas as pd #data frame operations
import numpy as np #arrays and math functions

#from scipy.stats import uniform #for training and test splits
#import statsmodels.api as sm  # statistical models (including regression)
#import statsmodels.formula.api as smf  # R-like model specification

import matplotlib.pyplot as plt #2D plotting
import seaborn as sns #seaborn for plotting
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator #for wordclouds
import folium #for maps
from folium import Marker #for maps
from folium.plugins import MarkerCluster #for maps
from folium import Choropleth
from folium.plugins import HeatMap
from folium import plugins

#from sklearn.linear_model import LinearRegression
#import plotly.graph_objs as go
#from plotly import tools
#from plotly.offline import iplot, init_notebook_mode

"""#Import Datasets"""

ufo = pd.read_csv("/content/newufo.csv")
air=pd.read_csv("/content/drive/MyDrive/Datasets/us-airports.csv")
mil=pd.read_csv("/content/drive/MyDrive/Datasets/US Military.csv")

#change datatype for the following columns
ufo["latitude"] = pd.to_numeric(ufo["latitude"], errors="coerce")
ufo['datetime'] = (pd.to_datetime(ufo['datetime'], format='%Y/%m/%d'))
ufo['date posted'] = (pd.to_datetime(ufo['date posted'], format='%Y/%m/%d'))
#add column day
ufo['day'] = pd.DatetimeIndex(ufo['datetime']).day
#drop unnecessary columns
cols2drop=[ 'Unnamed: 0','duration (hours/min)','date posted']
ufo=ufo.drop(columns=cols2drop)

print(air.type.unique())
#large_airport
air = air[air.type == 'large_airport']
air.shape

# Split coordinates into latitude and logitude
mil['Coordinates'] = mil['Coordinates'].apply(lambda x : x.split(sep = ','))
mil['latitude'] = mil['Coordinates'].apply(lambda x: x[0] )
mil['longtitude'] = mil['Coordinates'].apply(lambda x: x[1] )

"""#MAPS!"""

# Creating a function to visualise maps from a source of your choice
def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')

us_map = folium.Map(location=[40,-99], zoom_start=4)
mc = MarkerCluster()

# Adding marker clusters of US UFO sightings
for idx, row in ufo.iterrows():
        mc.add_child(folium.Marker([row['latitude'], row['longitude']]))
        
us_map.add_child(mc)
#embed_map(us_map, 'www.ufoclusters.html')
#us_map.save('ufoclusters.html')
us_map

us_map = folium.Map(location=[40,-99], zoom_start=4)
mc = MarkerCluster()

# Adding the airport markers to the map
for idx, row in air.iterrows():
    Marker([row['latitude_deg'], row['longitude_deg']]).add_to(us_map)

# Adding marker clusters of US UFO sightings
for idx, row in ufo.iterrows():
        mc.add_child(folium.Marker([row['latitude'], row['longitude']]))
        
us_map.add_child(mc)
#embed_map(us_map, 'www.ufoandairports.html')
#us_map.save('ufoandairports.html')
us_map

plt.subplots(figsize=(20,8))
#plt.hist(ufo['airport_dist'],edgecolor='black')
ax = sns.histplot(ufo['airport_dist'])
# Set the limit for each axis
plt.xlim(0, 20)
plt.ylim(0, 3000)
plt.title('UFO and Airports')
plt.ylabel('Number of UFO sightings')
plt.xlabel('Airport distance (kilometers)')

# Ensure you're handing it floats
mil['latitude'] = mil['latitude'].astype(float)
mil['longtitude'] = mil['longtitude'].astype(float)
us_map = folium.Map(location=[40,-99], zoom_start=4)
mc = MarkerCluster()

# Adding the military markers to the map
for idx, row in mil.iterrows():
    Marker([row['latitude'], row['longtitude']]).add_to(us_map)

# Adding marker clusters of US UFO sightings
for idx, row in ufo.iterrows():
        mc.add_child(folium.Marker([row['latitude'], row['longitude']]))
        
us_map.add_child(mc)
#embed_map(us_map, 'www.ufoandairports.html')
us_map.save('ufomilitary.html')
us_map

plt.subplots(figsize=(20,8))
ax = sns.histplot(ufo['military_base_dist'], palette="PuBuGn")
#plt.hist(ufo['military_base_dist'],edgecolor='black', bins=9)
plt.title('UFO and Military Bases')
plt.ylabel('Number of UFO sightings')
plt.xlabel('Militaty base distance (kilometers)')

dfmap = ufo.loc[:,['city','latitude','longitude']]
count = dfmap['city'].value_counts()
city = ufo['city']
count = pd.DataFrame(count,city)
count = count.rename(columns={' ': 'city','city': 'count'})
dfmap = dfmap.merge(count, on='city', how='left')
dfmap.drop_duplicates()

us_map = folium.Map(location=[40,-99], zoom_start=4)

# Ensure you're handing it floats
ufo['latitude'] = ufo['latitude'].astype(float)
ufo['longitude'] = ufo['longitude'].astype(float)

heat_df = ufo[['latitude', 'longitude']]
heat_df = heat_df.dropna(axis=0, subset=['latitude','longitude'])

# List comprehension to make out list of lists
heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(us_map)

# Display the map
us_map
#us_map.save('ufoheatmap.html')

sns.catplot(y=ufo["shape"], kind="count",
            palette="viridis", edgecolor=".6",
            data=ufo.sort_values("shape"))

ufo.loc[ufo['shape'] == 'rectangle', 'shape'] ='geometric'
ufo.loc[ufo['shape'] == 'chevron', 'shape'] ='geometric'
ufo.loc[ufo['shape'] == 'triangle', 'shape'] ='geometric'
ufo.loc[ufo['shape'] == 'diamond', 'shape'] ='geometric'
ufo.loc[ufo['shape'] == 'cross', 'shape'] ='geometric'
ufo.loc[ufo['shape'] == 'delta', 'shape'] ='geometric'
ufo.loc[ufo['shape'] == 'hexagon', 'shape'] ='geometric'
ufo.loc[ufo['shape'] == 'pyramid', 'shape'] ='geometric'

ufo.loc[ufo['shape'] == 'changed', 'shape'] ='changing'
ufo.loc[ufo['shape'] == 'formation','shape'] ='changing'

ufo.loc[ufo['shape'] == 'unknown','shape'] ='other'

ufo.loc[ufo['shape'] == 'flash', 'shape'] ='light'
ufo.loc[ufo['shape'] == 'flare', 'shape'] ='light'

ufo.loc[ufo['shape'] == 'circle', 'shape'] ='disk'
ufo.loc[ufo['shape'] == 'oval','shape'] ='disk'
ufo.loc[ufo['shape'] == 'round','shape'] ='disk'

ufo.loc[ufo['shape'] == 'spere', 'shape'] ='ball'
ufo.loc[ufo['shape'] == 'fireball', 'shape'] ='ball'

ufo.loc[ufo['shape'] == 'teardrop', 'shape'] ='egg'

ufo.loc[ufo['shape'] == 'cigar', 'shape'] ='cylinder'
ufo.loc[ufo['shape'] == 'cone','shape'] ='cylinder'
ufo.loc[ufo['shape'] == 'crescent','shape'] ='cylinder'

#ufo

s=sns.catplot(x=ufo["shape"], kind="count",
            color="royalblue", edgecolor=".6",
            data=ufo.sort_values("shape"))
s.set_xticklabels(rotation=70)
s.set_titles("Distribution of Shapes")

# Long sighting - light
df_light = ufo[ufo['shape'] == 'geometric']

us_map1 = folium.Map(location=[40,-99], zoom_start=4)
mc = MarkerCluster()

# Adding marker clusters of US UFO sightings
for idx, row in df_light.iterrows():
        mc.add_child(folium.Marker([row['latitude'], row['longitude']]))
        
us_map1.add_child(mc)
#us_map.save('ufoclusterlight.html')
us_map

us_map = folium.Map(location=[40,-99], zoom_start=4)

heat_df = df_light[['latitude', 'longitude']]
heat_df = heat_df.dropna(axis=0, subset=['latitude','longitude'])

# List comprehension to make out list of lists
heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(us_map)

# Display the map
us_map
#us_map.save('ufoheatmaplight2.html')

# Long sighting - egg
df_egg = ufo[ufo['shape'] == 'egg']

us_map = folium.Map(location=[40,-99], zoom_start=4)
mc = MarkerCluster()

# Adding marker clusters of US UFO sightings
for idx, row in df_egg.iterrows():
        mc.add_child(folium.Marker([row['latitude'], row['longitude']]))
        
us_map.add_child(mc)
#us_map.save('ufoclusteregg.html')
us_map

us_map = folium.Map(location=[40,-99], zoom_start=4)

heat_df = df_egg[['latitude', 'longitude']]
heat_df = heat_df.dropna(axis=0, subset=['latitude','longitude'])

# List comprehension to make out list of lists
heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(us_map)

# Display the map
#us_map
#us_map.save('ufoheatmapegg.html')
us_map

ufoa = ufo[ufo["duration (seconds)"] <= 4000]
ufo.describe()

plt.subplots(figsize=(10,8))
plt.hist(ufoa['duration (seconds)'])
plt.title('Duration Distribution')
plt.ylabel('Number of UFO sightings')
plt.xlabel('Duration (seconds)')

plt.subplots(figsize=(10,8))
ax = sns.histplot(ufoa['duration (seconds)'])
# Set the limit for each axis
#plt.xlim(0, 20)
#plt.ylim(0, 3000)
plt.title('Duration Distribution')
plt.ylabel('Number of UFO sightings')
plt.xlabel('Duration (seconds)')

ufob = ufo[ufo["duration (seconds)"] >= 86400]

us_map = folium.Map(location=[40,-99], zoom_start=4)

heat_df = ufob[['latitude', 'longitude']]
heat_df = heat_df.dropna(axis=0, subset=['latitude','longitude'])

# List comprehension to make out list of lists
heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(us_map)

# Display the map
us_map
#us_map.save('ufo24h.html')