# Databricks notebook source
# MAGIC %md 
# MAGIC # Challenge Databricks x Microsoft: Wind Turbine Anomaly Detection
# MAGIC # Part 1 : Data cleaning
# MAGIC 
# MAGIC **Authors**
# MAGIC - Amine HADJ-YOUCEF
# MAGIC - Maxime  CONVERT
# MAGIC - Cassandre DIAINE
# MAGIC - Axel DIDIER

# COMMAND ----------

# MAGIC %md
# MAGIC ![Turbine Diagram](https://www.researchgate.net/profile/Marcias-Martinez/publication/265382474/figure/fig1/AS:392207954661392@1470521071896/Wind-turbine-schematic-source-US-DOE.png)
# MAGIC 
# MAGIC - [Get the data here](https://opendata-renewables.engie.com/explore/dataset/01c55756-5cd6-4f60-9f63-2d771bb25a1a/information#)
# MAGIC - [How a Wind Turbine Works - Text Version](https://www.energy.gov/eere/wind/how-wind-turbine-works-text-version)

# COMMAND ----------

# MAGIC %md
# MAGIC <!-- <strong>Discussion:</strong> 
# MAGIC  -->
# MAGIC We first load the data and split it between the four turbines to make the experimentation clearer and easier to handle in the context of this datathon. 
# MAGIC 
# MAGIC This notebook is organized to exlore and present the given data. We therefore handle the four turbines individually to get a good feel of what is available. 
# MAGIC 
# MAGIC In an actual production scenario where we would potentially deal with tens or hundreds of wind turbines, we would reorganize the code to fully make use of parallelization instead of exploring each turbine one by one.
# MAGIC 
# MAGIC We also perform initial data manipulation by converting all numerical features to floats, and by getting rid of the turbine name in each dataframe to allow comparisons in further steps.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading data

# COMMAND ----------

# Library imports
import pyspark.pandas as ps # pandas on spark
import pandas as pd
import numpy as np
import plotly
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ps.options.display.max_rows = 10
ps.set_option('compute.ops_on_diff_frames', True)

# COMMAND ----------

# Loading data
sql_table = "hackaton_team_sncf.la_haute_borne_data_bronze"
df = spark.read.table(sql_table).to_pandas_on_spark() # koalas
cols_feat = df.columns[~df.columns.isin(["Wind_turbine_name", "Date_time"])].tolist()

# Loading data descriptions
data_desc_path = "/dbfs/FileStore/shared_uploads/pahy04211@commun.ad.sncf.fr/data_description.csv"
df_desc = pd.read_csv(data_desc_path, sep=";")

# COMMAND ----------

df_desc

# COMMAND ----------

print(cols_feat)

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# Turbine names
turbine_names = ["R80711","R80721", "R80736", "R80790"]

# DataFrames for each turbine
df1s = df[df['Wind_turbine_name'] == turbine_names[0]].copy()
df2s = df[df['Wind_turbine_name'] == turbine_names[1]].copy()
df3s = df[df['Wind_turbine_name'] == turbine_names[2]].copy()
df4s = df[df['Wind_turbine_name'] == turbine_names[3]].copy()

# Set all dataframes to a dictionary 
df_turbine = {"R80711": df1s, "R80721": df2s, "R80736": df3s, "R80790":df4s} 

# Dropping column "Wind_turbine_name" from each dataframe 
for turbine in turbine_names:
  df_turbine[turbine] = df_turbine[turbine].drop(columns=['Wind_turbine_name'], axis=1).sort_values(by=["Date_time"])

# COMMAND ----------

print("Dataframe dimensions :")
for turbine in turbine_names:
   print("Turbine {} - {}".format(turbine, df_turbine[turbine].shape))
    

# COMMAND ----------

# Convert feature data types to float
dict_type={}
for i in df_turbine["R80711"].columns[1:]:
  dict_type[i]='float'

for turbine in turbine_names:
  df_turbine[turbine] = df_turbine[turbine].astype(dict_type) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dropping empty rows for each turbine

# COMMAND ----------

print("Dataframe dimensions :")
print("Before :")
for turbine in turbine_names:
   print("Turbine {} - {}".format(turbine, df_turbine[turbine].shape))

for turbine in turbine_names:
  df_turbine[turbine] = df_turbine[turbine].dropna(how="all", axis=0)

print("After :")
for turbine in turbine_names:
   print("Turbine {} - {}".format(turbine, df_turbine[turbine].shape))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Dropping timestamp duplicates

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Discussion:</strong> 
# MAGIC 
# MAGIC The data for each turbine contains timestamp duplicates. Some of these correspond to rows with all NaNs which were removed in the previous step, and others seem to contain valid numerical values but different across the duplicate timestamps. We decide to keep the first occurrence of the two duplicates and drop the second one, based on the assumption that the second duplicate corresponds to a delay in the sensors to send the data to the logger, and the first timestamp should contain all the relevant information. Knowing more on the data acquisition process would certainly help in better addressing this question in an ideal scenario.

# COMMAND ----------

# https://databricks.com/blog/2020/03/31/10-minutes-from-pandas-to-koalas-on-apache-spark.html
print("Dataframe dimensions :")
print("Before :")
for turbine in turbine_names:
   print("Turbine {} - {}".format(turbine, df_turbine[turbine].shape))
    
for turbine in turbine_names:
  df_turbine[turbine] = df_turbine[turbine][~df_turbine[turbine].duplicated(subset='Date_time', keep='first')]

print("After :")
for turbine in turbine_names:
   print("Turbine {} - {}".format(turbine, df_turbine[turbine].shape))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filtering missing values

# COMMAND ----------

get_df_nan_pct = lambda x: x.isna().sum()*100/x.shape[0]

df_turbine_nan = {}

for turbine in turbine_names:
  df_turbine_nan[turbine] = get_df_nan_pct(df_turbine[turbine])

# COMMAND ----------

# Get column names containing null values
nan_pct = 2
df_cols_nan = {}
for turbine in turbine_names:
  df_cols_nan[turbine] = df_turbine_nan[turbine].loc[df_turbine_nan[turbine]>nan_pct].index.tolist()

# COMMAND ----------

# Display sensors corresponding to the columns with missing values
get_var_name = lambda x: df_desc.loc[df_desc.Variable_name.isin(x), "Variable_long_name"].values.tolist()

df_featname_nan = {}
for turbine in turbine_names:
  df_featname_nan[turbine] = pd.Series([x[:-4]  for x in df_cols_nan[turbine]]).unique()

print(f"Sensors returning NaNs: \n")
for turbine in turbine_names:
    print("Turbine {}: {} ".format(turbine, get_var_name(df_featname_nan[turbine])))

# COMMAND ----------

# Dropping columns containing missing values

for turbine in turbine_names:
  df_turbine[turbine] = df_turbine[turbine].drop(columns=df_cols_nan[turbine])


# COMMAND ----------

print("Number of remaining features:")
for turbine in turbine_names:
  featname_restant = pd.Series([x[:-4]  for x in df_turbine[turbine].columns]).unique()[2:]
  print(f"Turbine {turbine} : {len(df_turbine[turbine].columns)}, corresponds to {featname_restant.shape[0]} sensors: \n {featname_restant}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Handling missing timestamps

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Discussion:</strong>
# MAGIC 
# MAGIC Some timestamps are missing from the data. We discussed whether filling those spots would be helpful, so we first checked whether the missing timestamps were sparse events that could easily be fillable, or if they corresponded to extended periods of time. We observed the presence of some missing data stretching over several days. As our end goal was to build a rather simple model by focusing on point anomaly detection methods instead of working with time series, we decided to leave these missing timestamps untouched as they wouldn't prevent the work in the next steps.

# COMMAND ----------

print("Dataset dimensions :")
for turbine in turbine_names:
   print("Turbine {} - {} : {} unique timestamps ".format(turbine, df_turbine[turbine].shape, len(df_turbine[turbine]['Date_time'].unique())))


# COMMAND ----------

print("Timestamp intervals: \n ")
for turbine in turbine_names:
   print("Turbine {} - Start {} - End {}  ".format(turbine, df_turbine[turbine]['Date_time'].min(), df_turbine[turbine]['Date_time'].max()))

# COMMAND ----------

# Check whether missing timestamps are sparse or extend over periods
def check_missing_periods(df_ts):
  periods = []
  count = 0
  for i in range(1, len(df_ts)):
    if (df_ts[i] - pd.Timedelta(minutes=10) == df_ts[i-1]):
      count += 1
    else:
      count += 1
      periods.append(count)
      count = 0
  return periods

# Generate the actual timestamps and compare with the ones in the data
t_range = pd.date_range(start='31-12-2012 23:00:00', end='12-01-2018 23:00:00', freq='10T')

df_missing = {}
df_missing_periods = {}
for turbine in turbine_names:
  df_missing[turbine] = t_range.difference(df_turbine[turbine]['Date_time'].values)
  df_missing_periods[turbine] = check_missing_periods(df_missing[turbine])  
  

# COMMAND ----------

for turbine in turbine_names:
  print(sorted(df_missing_periods[turbine], reverse=True)[:6])

# COMMAND ----------

# MAGIC %md
# MAGIC The longest period of consecutively missing timestamps amounts to 797*10=7970 minutes => 5.53 days.

# COMMAND ----------

# print(df1_missing.shape, df2_missing.shape, df3_missing.shape, df4_missing.shape)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Filling missing values

# COMMAND ----------

# MAGIC %md
# MAGIC We decided to fill the missing values by propagating the last preceding value for each variable to maintain continuity in the data, as other methods (e.g. filling with median value) may potentially generate unforeseen anomalous scenarios. With the used filling method, if an anomaly occurs prior to a missing value, the anomaly would simply be propagated and could still be detected by an anomaly detection method.

# COMMAND ----------

for turbine in turbine_names:
  df_turbine[turbine] = df_turbine[turbine].fillna(method='ffill')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature engineering

# COMMAND ----------


for turbine in turbine_names:
  # Weekly moving average
  df_turbine[turbine]['mavg_Gb1t_avg_7d'] = df_turbine[turbine]['Gb1t_avg'].rolling(window=1008).mean()
  # Gearbox temperature
  df_turbine[turbine]["gb1_outside_avg"] = (df_turbine[turbine]['Gb1t_avg'] - df_turbine[turbine]['Ot_avg']).abs() 
  # Time-related categorical variables
  df_turbine[turbine]['year'] = df_turbine[turbine]['Date_time'].dt.year
  df_turbine[turbine]['month'] = df_turbine[turbine]['Date_time'].dt.month
  df_turbine[turbine]['hourly'] = df_turbine[turbine]['Date_time'].dt.hour


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exporting cleaned data

# COMMAND ----------

for turbine in turbine_names:
  print("Saving turbine data ", turbine)
  df_turbine[turbine].to_spark().write.format('delta').mode('overwrite').saveAsTable('hackaton_team_sncf.df_post_turbine_'+turbine)

# COMMAND ----------

# Concatenate datasets to feed to the PowerBI dashboard
# df1s['Wind_turbine_name'] = "R80711"
# df2s['Wind_turbine_name'] = "R80721"
# df3s['Wind_turbine_name'] = "R80736"
# df4s['Wind_turbine_name'] = "R80790"

# df_concat = ps.concat([df1s[df1s.columns], df2s[df1s.columns], df3s[df1s.columns], df4s[df1s.columns]], axis=0)
# df_concat.to_spark().write.format('delta').mode('overwrite').saveAsTable('hackaton_team_sncf.df_post_turbine_all')

# COMMAND ----------


