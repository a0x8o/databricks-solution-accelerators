# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Challenge Databricks x Microsoft: Wind Turbine Anomaly Detection
# MAGIC # Part 2 : Data exploration
# MAGIC 
# MAGIC **Authors**
# MAGIC - Amine HADJ-YOUCEF
# MAGIC - Maxime  CONVERT
# MAGIC - Cassandre DIAINE
# MAGIC - Axel DIDIER

# COMMAND ----------

# MAGIC %md 
# MAGIC In this notebook we perform data exploration, with multi-univariate analysis, in addition to visualization

# COMMAND ----------

import pandas as pd
import pyspark.pandas as pd_sp # pandas on spark
import pandas as pd
import numpy as np
import plotly
import os

import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql.types import *

import seaborn as sns

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go 

from collections import Counter
sns.set_theme(style="white")

# COMMAND ----------

# MAGIC %md
# MAGIC # Raw dataset
# MAGIC 
# MAGIC First part of the data visualization is made on the raw dataset.

# COMMAND ----------

# Import data
dataset = spark.read.table("hackaton_team_sncf.la_haute_borne_data_bronze")
dataset_pdsp=dataset.to_pandas_on_spark()

# COMMAND ----------

# df_vis = spark.read.table('hackaton_team_sncf.df_post_turbine_R80721').to_pandas_on_spark().dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search for seasonality in wind turbine sensors
# MAGIC 
# MAGIC The visualisation is done on the data from turbine R80721 during one month in June 2016.
# MAGIC 
# MAGIC Focus on sensors : Ds, Gb1t_avg, Gb2t_avg, Ot_avg, Git_avg, Ws_avg, Rs_avg, P_avg.

# COMMAND ----------

# Data during one month in June 2016
data_R80721 = dataset_pdsp.query('(Date_time > "2016-06-01" and Date_time < "2016-06-30") and Wind_turbine_name == "R80721"')
                        
col_keep = ['Wind_turbine_name', 'Date_time', 'Gb1t_avg', 'Gb2t_avg', 'Ot_avg', 'Git_avg', 'Ws_avg', 'Rs_avg', 'P_avg', 'Ds_avg']
data_R80721 = data_R80721[col_keep]
data_R80721 = data_R80721.to_pandas() 

# COMMAND ----------

# MAGIC %md
# MAGIC ###### External factors (outdoor temperature and wind speed)

# COMMAND ----------

# MAGIC %md
# MAGIC A first idea was to look at the behavior of external parameters to the wind turbine such as wind and outside temperature to see how much these two parameters can be correlated.

# COMMAND ----------

# Outdoor temperature and wind speed
plt.figure(figsize=(25,4))
sns.lineplot(data=data_R80721, x='Date_time', y='Ot_avg', label="Ot_avg")
sns.lineplot(data=data_R80721, x='Date_time', y='Ws_avg', label="Ws_avg")
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC As one might have expected, there seems to be no correlation between wind and temperature

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Gearbox bearing temperature
# MAGIC 
# MAGIC The next step was to examine the characteristics concerning the temperature of the gearbox and to compare their values with each other and with the outside temperature.

# COMMAND ----------

plt.figure(figsize=(25,5))
plt.legend()
sns.lineplot(x='Date_time', y='Gb1t_avg', data=data_R80721, label="Gb1t_avg")
sns.lineplot(x='Date_time', y='Gb2t_avg', data=data_R80721, label="Gb2t_avg")
sns.lineplot(x='Date_time', y='Git_avg', data=data_R80721, label="Git_avg")
sns.lineplot(x='Date_time', y='Ot_avg', data=data_R80721, label="Ot_avg")


# COMMAND ----------

# MAGIC %md
# MAGIC Logically the Gbt1 and Gbt2 sensors have the same values, so if their behaviors are different it means that there may be an anomaly. The Git sensor has the same behavior as the Gbt1 and 2 sensors even though the high values are lower.

# COMMAND ----------

# Generator speed
plt.figure(figsize=(25,4))
plt.subplot(2, 1, 1)
sns.lineplot(x='Date_time', y='Rs_avg', data=data_R80721, label='Rs_avg')
plt.subplot(2, 1, 2)
sns.lineplot(x='Date_time', y='Ds_avg', data=data_R80721, label='Ds_avg' , color='r')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The behavior of the sensors Ds_avg (generator speed) and Rs_avg(rotor speed) is similar, only the scale differs. There seems to be a factor 100 between the two features. The rotor drives the generator so logically the behavior of their sensors is similar. Again, a desynchronization of the two sensors would be a sign of an anomaly.

# COMMAND ----------

# MAGIC %md
# MAGIC #### P_avg = Active_power (kW)

# COMMAND ----------

plt.figure(figsize=(25,5))
plt.subplot(2, 1, 1)
sns.lineplot(data=data_R80721, x='Date_time', y='P_avg', hue='Wind_turbine_name')
plt.subplot(2, 1, 2)
sns.lineplot(data=data_R80721, x='Date_time', y='Ws_avg', label="Ws_avg", color='r')


# COMMAND ----------

# MAGIC %md
# MAGIC As expected, wind and active power are highly correlated. It takes a lot of wind to drive the turbine and thus have production. Production peaks are visible and coincide with wind peaks. Thus extreme production values do not seem to be directly related to anomalies.

# COMMAND ----------

# MAGIC %md
# MAGIC ####  On all these graphs, there is no obvious seasonality. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search for disparity between the turbines.
# MAGIC Can we make one model for all turbines or one model per turbine ?
# MAGIC 
# MAGIC 
# MAGIC Focus on data from the 4 turbines during one week in June 2016
# MAGIC 
# MAGIC 
# MAGIC Focus on features Ds_avg, Gb1t_avg, Gb2t_avg, Ot_avg, Git_avg, Ws_avg, Rs_avg, P_avg

# COMMAND ----------

data = dataset_pdsp.query('(Date_time > "2016-06-01" and Date_time < "2016-06-07")')
                        
col_keep = ['Wind_turbine_name', 'Date_time', 'Gb1t_avg', 'Gb2t_avg', 'Ot_avg', 'Git_avg', 'Ws_avg',
            'Rs_avg', 'P_avg', 'Ds_avg']

data = data[col_keep]
data = data.to_pandas() 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparison of sensor values for the four turbines
# MAGIC 
# MAGIC ##### Internal factors

# COMMAND ----------

plt.figure(figsize=(25,20))
plt.subplot(6, 1, 1)
sns.lineplot(data=data, x='Date_time', y='Gb1t_avg', hue='Wind_turbine_name')
plt.subplot(6, 1, 2)
sns.lineplot(data=data, x='Date_time', y='Gb2t_avg', hue='Wind_turbine_name')
plt.subplot(6, 1, 3)
sns.lineplot(data=data, x='Date_time', y='Git_avg', hue='Wind_turbine_name')
plt.subplot(6, 1, 4)
sns.lineplot(data=data, x='Date_time', y='Ds_avg', hue='Wind_turbine_name')
plt.subplot(6, 1, 5)
sns.lineplot(data=data, x='Date_time', y='Rs_avg', hue='Wind_turbine_name')
plt.subplot(6, 1, 6)
sns.lineplot(data=data, x='Date_time', y='P_avg', hue='Wind_turbine_name')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### External factors

# COMMAND ----------

plt.figure(figsize=(25,8))
plt.subplot(2, 1, 1)
sns.lineplot(data=data, x='Date_time', y='Ot_avg',  hue='Wind_turbine_name')
plt.subplot(2, 1, 2)
sns.lineplot(data=data, x='Date_time', y='Ws_avg', hue='Wind_turbine_name')

# COMMAND ----------

# MAGIC %md
# MAGIC There are no major differences in the behavior of the four turbines. For the external data common to the four turbines such as wind or outdoor temperature, very slight differences are observable when they should be the same. Given the volume of data, it seems more interesting to make a model per turbine so that they are as personalized as possible. Moreover, no information is given concerning the type of material, their age, if they have the same brand,... which can impact on the production of wind turbines. A model by turbine also allows to compensate for the lack of data concerning the material.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Active power comparison according to wind speed
# MAGIC 
# MAGIC While doing some research on the subject of wind turbines, it appeared that wind turbine panes only rotate if the wind speed is at least 10 km/h (between 10 and 15km/h according to the articles). It seemed interesting to compare the output of the wind turbine with the wind speed. 

# COMMAND ----------

# Create a new binary feature Ws_inf10 : 1 if the maximum wind speed during the 10 minutes is inferior to 10 km/h else 0. 
# So it means if Ws_inf10==1, the wind turbine is not supposed to turn.
dataset_pdsp.loc[:, "Ws_inf10"] = dataset_pdsp.Ws_max.apply(lambda x : 1 if x < 10 else 0)
data_wind_viz = dataset_pdsp[['Ws_inf10', 'P_avg']]
data_wind_viz = data_wind_viz.to_pandas()
fig = data_wind_viz.boxplot(by='Ws_inf10', return_type='axes')
fig

# COMMAND ----------

# 1 means windspeed<10km/h, else 0
Counter(data_wind_viz.Ws_inf10)

# COMMAND ----------

# MAGIC %md
# MAGIC On 80% of the data the wind is less than 10 km/h

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation matrix
# MAGIC 
# MAGIC Here we plot the correlation matrix by focusing only on the avg value of each feature.

# COMMAND ----------

# Drop features with more than 1% of missing values
col_dataset = dataset_pdsp.columns.to_list()
col_va =  [col for col in col_dataset if col.startswith('Va')]
col_wa =  [col for col in col_dataset if col.startswith('Wa_c')]
col_na =  [col for col in col_dataset if col.startswith('Na_c')]
col_pas =  [col for col in col_dataset if col.startswith('Pas')]

col_drop = col_va + col_wa +col_na + col_pas
data_red = dataset_pdsp.drop(col_drop, axis=1)

# COMMAND ----------

col_keep_avg = [col for col in data_red.columns.to_list() if col.endswith('_avg')]
data_red = data_red[col_keep_avg]
data_red = data_red.dropna().reset_index(drop=True)
data_red = data_red.to_pandas()

# COMMAND ----------

corr_df = data_red.corr(method='pearson')

plt.figure(figsize=(8, 8))
plt.title("Features Correlation")
sns.heatmap(corr_df)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As can be seen from the correlation matrix many variables are highly correlated, which illustrates what was previously highlighted in the charts. The fact that there is a lot of correlation can help to detect anomalies because the behaviors of the sensors are very related to each other. The indoor and outdoor temperature are not correlated contrary to our intuition.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Day/night impact on electricity production at the turbine outlet
# MAGIC According to our research, the density of the air is higher at night, which makes the blades of the wind turbine turn faster. However, other articles point out that at night the wind drops. The interest is to see roughly if the night has an impact on the output of the wind turbine.

# COMMAND ----------

dataset_pdsp.loc[:, "hour"] = dataset_pdsp.Date_time.apply(lambda x: x.hour)
dataset_pdsp.loc[:, "month"] = dataset_pdsp.Date_time.apply(lambda x: x.month)
dataset_pdsp.loc[:, "date"] = dataset_pdsp.Date_time.apply(lambda x: x.strftime('%Y-%m-%d'))
dataset_pdsp.date = dataset_pdsp.date
dataset_pdsp.loc[:, "night"] = dataset_pdsp.hour.apply(lambda x: 0 if (x>=7 and x<=20) else 1)

data_R80711 = dataset_pdsp[dataset_pdsp.Wind_turbine_name == 'R80711']

data_R80711_pandas = data_R80711.to_pandas()

boxplot_day_night = sns.boxplot(x="month", y="P_avg", hue="night",
                                data=data_R80711_pandas, palette="Set3")
boxplot_day_night


# COMMAND ----------

# Plot sum P_avg producted day/night, 6 months of data for wind turbine R80711
data_R80711_mean_day_night = data_R80711.groupby(['date', 'night'])['P_avg'].sum().reset_index()
plot_R80711_night = data_R80711_mean_day_night.query('(date > "2016-01-01" and date < "2016-06-30")')
plot_R80711_night = plot_R80711_night.to_pandas()

plt.figure(figsize=(25,8))
sns.lineplot(data=plot_R80711_night, x='date', y='P_avg', hue='night')

# COMMAND ----------

# MAGIC %md
# MAGIC At night the production is always a little higher than during the day. This boxplot also highlights the presence of seasonality but on a yearly scale of electricity production

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing values
# MAGIC 
# MAGIC The next step was to look at the missing values in dataset (how much ? on which features ?) in order to deal with it

# COMMAND ----------

get_df_nan_pct = lambda x: x.isna().sum()*100/x.shape[0]

df_nan = get_df_nan_pct(dataset_pdsp).to_pandas()
fig = go.Figure([go.Bar(x=df_nan.index, 
                         y=df_nan.values)])
fig.update_layout( yaxis_title="NaN [%]",title='Distribution of missing values', height=400, width=700)
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC As you can see it on the plot, six features were problematic : Na_c, Wa_c, Pas, Va, Va1 and Va2. The choice was made to remove features which have more than 2% of missig values. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Values taken by each feature
# MAGIC 
# MAGIC This part of dataviz is made on clean dataset. The cleaning is made in the notebook 1_Cleaning. The idea was to look at the distribution of each feature.

# COMMAND ----------

df1s = spark.read.table('hackaton_team_sncf.df_post_turbine_r80711').to_pandas_on_spark()
df2s = spark.read.table('hackaton_team_sncf.df_post_turbine_r80721').to_pandas_on_spark()
df3s = spark.read.table('hackaton_team_sncf.df_post_turbine_r80736').to_pandas_on_spark()
df4s = spark.read.table('hackaton_team_sncf.df_post_turbine_r80790').to_pandas_on_spark()

# COMMAND ----------

# Focus on 10 000 rows of the dataframe of turbine R80711 selected randomly due to memory problems
df1_pd = df1s.to_pandas()
df1_pd_sample = df1_pd.sample(n = 10000)

lst_col_avg = [col for col in df1_pd_sample.columns if col.endswith('_avg')]
df1_pd_sample_red = df1_pd_sample[lst_col_avg]

fig = go.Figure()
for i in df1_pd_sample_red.columns:
    fig.add_trace(go.Box(y=df1_pd_sample_red[i], name=i))
fig.update_layout(height=800, legend=dict(orientation="h"))
fig.show()


# COMMAND ----------

# Focus on 10 000 rows of the dataframe of turbine R80721 selected randomly due to memory problems
df2_pd = df2s.to_pandas()
df2_pd_sample = df2_pd.sample(n = 10000)

lst_col_avg = [col for col in df2_pd_sample.columns if col.endswith('_avg')]
df2_pd_sample_red = df2_pd_sample[lst_col_avg]

fig = go.Figure()
for i in df2_pd_sample_red.columns:
    fig.add_trace(go.Box(y=df2_pd_sample_red[i], name=i))
fig.update_layout(height=800, legend=dict(orientation="h"))
fig.show()


# COMMAND ----------

# Focus on 10 000 rows of the dataframe of turbine R80736 selected randomly due to memory problems
df2_pd = df2s.to_pandas()
df2_pd_sample = df2_pd.sample(n = 10000)

lst_col_avg = [col for col in df2_pd_sample.columns if col.endswith('_avg')]
df2_pd_sample_red = df2_pd_sample[lst_col_avg]

fig = go.Figure()
for i in df2_pd_sample_red.columns:
    fig.add_trace(go.Box(y=df2_pd_sample_red[i], name=i))
fig.update_layout(height=800, legend=dict(orientation="h"))
fig.show()


# COMMAND ----------

# Focus on 10 000 rows of the dataframe of turbine R80790 selected randomly due to memory problems
df4_pd = df4s.to_pandas()
df4_pd_sample = df4_pd.sample(n = 10000)

lst_col_avg = [col for col in df4_pd_sample.columns if col.endswith('_avg')]
df4_pd_sample_red = df4_pd_sample[lst_col_avg]

fig = go.Figure()
for i in df4_pd_sample_red.columns:
    fig.add_trace(go.Box(y=df4_pd_sample_red[i], name=i))
fig.update_layout(height=800, legend=dict(orientation="h"))
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC As you can see on this plot, we can already see some outliers for some features such as Cm, DCs, ...

# COMMAND ----------


