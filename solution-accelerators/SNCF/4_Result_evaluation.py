# Databricks notebook source
# MAGIC %md
# MAGIC # Challenge Databricks x Microsoft: Wind Turbine Anomaly Detection
# MAGIC # Part 4 : Result Evaluation
# MAGIC 
# MAGIC **Authors**
# MAGIC - Amine HADJ-YOUCEF
# MAGIC - Maxime  CONVERT
# MAGIC - Cassandre DIAINE
# MAGIC - Axel DIDIER

# COMMAND ----------

import pandas as pd
import pyspark.pandas as pd_sp
import numpy as np
import plotly
import os
import plotly.graph_objects as go 
from collections import Counter

import mlflow, time

from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import warnings
pd.options.mode.chained_assignment = None
pd_sp.set_option('compute.ops_on_diff_frames', True)

# COMMAND ----------

turbine = "r80711"
dfiforest1 = spark.read.table('hackaton_team_sncf.df_res_iforest_'+turbine).to_pandas_on_spark().dropna()
dflof1 = spark.read.table('hackaton_team_sncf.df_res_lof_'+turbine).to_pandas_on_spark().dropna()
dfclass1 = spark.read.table('hackaton_team_sncf.df_res_class_'+turbine).to_pandas_on_spark().dropna()

# COMMAND ----------

# label column outlier_iforest : 1 if outlier, else 0
dfiforest1.outlier_iforest = dfiforest1.outlier_iforest.apply(lambda x: 1 if x==-1 else 0)
dflof1.outlier_lof = dflof1.outlier_lof.apply(lambda x: 1 if x==-1 else 0)

# COMMAND ----------

# MAGIC %md
# MAGIC # Result Visualisation

# COMMAND ----------

# MAGIC %md
# MAGIC To get a sense of the model, we propose to visualize the outliers returned by two algorithms: Local Outlier Factor and Isolation Forest. 
# MAGIC 
# MAGIC We organize the discussion of different variables according to their physical position and interactions within the turbines.

# COMMAND ----------

def plot_result(turbine, feat, tmp1, tmp2):
  """
  This function plot features and highlights outlier detected using two methods
  """
  method1 = 'LocalOutlierFactor'
  method2 = 'IsolationForest'

  fig = make_subplots(rows=1, cols=2, subplot_titles=[method1, method2], shared_xaxes=True)

  fig.add_trace(go.Scatter(x=tmp1["Date_time"], y=tmp1[feat], name='normal', mode='markers'), row=1, col=1)
  fig.add_trace(go.Scatter(x=tmp1.loc[tmp1.outlier_lof==1, "Date_time"], y=tmp1.loc[tmp1.outlier_lof==1, feat], name='outlier_'+method1, mode='markers'),row=1, col=1)

  fig.add_trace(go.Scatter(x=tmp2["Date_time"], y=tmp2[feat], name='normal', mode='markers'), row=1, col=2)
  fig.add_trace(go.Scatter(x=tmp2.loc[tmp2.outlier_iforest==1, "Date_time"], y=tmp2.loc[tmp2.outlier_iforest==1, feat], name='outlier_'+method2, mode='markers'), row=1, col=2)


  fig.update_layout(height=500, width=1800, title_text=turbine+feat)
  fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ![Turbine Diagram](https://www.researchgate.net/profile/Marcias-Martinez/publication/265382474/figure/fig1/AS:392207954661392@1470521071896/Wind-turbine-schematic-source-US-DOE.png)

# COMMAND ----------

# We chose to select a sample of data to visualize results and focus on a single turbine R80711
tmp1 = dflof1.sample(frac=.03, random_state=42).to_pandas()
tmp2 = dfiforest1.sample(frac=.03, random_state=42).to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## External Temperature 

# COMMAND ----------

feat = 'Ot_avg'
plot_result("R80711: ", feat, tmp1, tmp2)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Discussion</strong>: From the above plot, it seems like outside temperature is not a good predictor of abnormal values. We saw in the data exploration that the outside temperature is not especially correlated to any other variables of the inner system of the turbine, suggesting we should look for anomalies within the turbine itself and the interaction between its different components.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wind

# COMMAND ----------

plot_result("R80711: ", 'Wa_avg', tmp1, tmp2)

# COMMAND ----------

plot_result("R80711: ", 'Ws_avg', tmp1, tmp2)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Discussion:</strong> Anomalies do not seem to be linked to the wind direction, which is not surprising considering the turbines should follow the wind direction and work optimally regardless of the wind direction, and there might be minimal anomalies occurring in the outer structure of the wind turbines. As for the wind speed, the anomalies found by Local Outlier Factor do not appear correlated to high or low wind speed. On the other hand, Isolation Forest seems to especially highlight values occurring with unusually high wind speed. From these observations, we can interpret that either LOF performance is poorer than Isolation Forest, or that it is simply finding outliers that are not directly linked to wind speed, unlike Isolation Forest, which seems to heavily take acccount of that variable in its classification.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gearbox

# COMMAND ----------

plot_result("R80711: ", 'Gb1t_avg', tmp1, tmp2)

# COMMAND ----------

plot_result("R80711: ", 'Gb2t_avg', tmp1, tmp2)

# COMMAND ----------

plot_result("R80711: ", 'Gost_avg', tmp1, tmp2)

# COMMAND ----------

plot_result("R80711: ", 'Git_avg', tmp1, tmp2)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Discussion</strong>: We saw in the correlation matrix that the temperatures for gearbox 1 and 2 are heavily correlated, and here, Isolation Forest highlights gearbox temperatures unusually low, as well as unusually high as we can see for data points towardds the end of 2017. Local outlier Factor does not seem to highlight any obvious pattern though. Nevertheless, there still are some anomalies occurring under nominal gearbox temperature values, which suggests Isolation Forest detects other types of anomalies as well. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Power

# COMMAND ----------

plot_result("R80711: ", "P_avg", tmp1, tmp2)

# COMMAND ----------

plot_result("R80711: ", "S_avg", tmp1, tmp2)

# COMMAND ----------

plot_result("R80711: ", "Q_avg", tmp1, tmp2)

# COMMAND ----------

# MAGIC %md
# MAGIC <strong>Discussion:</strong> Here we see, Isolation Forest highlighting unusually high levels of power production, as well as those close to zero. With further exploration by conducting more thorough multivariate analysis, we may be able to identify clearer root causes around these anomalies, where the turbine may be underperforming despite normal operating conditions, but we would need to do so with better visualization tools and other statistical methods.

# COMMAND ----------

# MAGIC %md
# MAGIC # Earnings estimate: Turbine R80711

# COMMAND ----------

# MAGIC %md
# MAGIC In order to evaluate the developed model we focus on one of the four turbines (turbine R80711). Ideally, the results discussed below would be applied to all other turbines in the farm. The choice is made to consider isolated forest as the best model to detect anomalies.
# MAGIC 
# MAGIC 
# MAGIC Anomalies can lead to a decrease in the production of wind turbines. The idea here is to roughly quantify this loss and the loss of profit for the company.

# COMMAND ----------

# MAGIC %md
# MAGIC The problem with unsupervised algorithms is that it is not possible to know if the outlier detection is good insight from field expert. To evaluate the earning, the idea is to do a basic AutoML quickely to predict the P_avg. To do that, the rows which are considered outlier by at least one method are removed. 

# COMMAND ----------

col_iforest = ['Date_time', 'outlier_iforest']
dfiforest1_red = dfiforest1[col_iforest]
col_lof = ['Date_time', 'outlier_lof']
dflof1_red = dflof1[col_lof]
col_class = ['Date_time', 'outlier']
dfclass1_red = dfclass1[col_class]

# Summary of outliers per method : classic, lof and iforest
df_ifor_lof1 = dfiforest1_red.merge(dflof1_red, how='left', left_on='Date_time', right_on='Date_time')
df_ifor_lof_class1 = dfclass1_red.merge(df_ifor_lof1, how='left', left_on='Date_time', right_on='Date_time')
df_ifor_lof_class1 = df_ifor_lof_class1.to_pandas()
lst = ['outlier_iforest', 'outlier_lof', 'outlier']
df_ifor_lof_class1.loc[:, "sum_outlier"] =  df_ifor_lof_class1[lst].sum(axis = 1)
df_ifor_lof_class1.loc[:, "outlier_any_method"] =  df_ifor_lof_class1.sum_outlier.apply(lambda x: 1 if x>0 else 0)

# Keep only row which are never consider as outlier
dfiforest1_pd = dfiforest1.to_pandas()
data_P_avg = dfiforest1_pd.merge(df_ifor_lof_class1, how='left', left_on='Date_time', right_on='Date_time')
data_P_avg = data_P_avg[data_P_avg.outlier_any_method==0]
data_P_avg_sp=spark.createDataFrame(data_P_avg) 

#data_P_avg_sp.write.format('delta').mode('overwrite').saveAsTable('hackaton_team_sncf.data_P_avg_autoML_R80711')

# COMMAND ----------

# MAGIC %md
# MAGIC For AutoML, all the variables concerning Q and S are deleted because P is a combination of these two characteristics. The variable to predict and P_avg, the variables P_min, P_max, ... are also logically deleted.
# MAGIC 
# MAGIC 
# MAGIC Once the AutoML has finished running, the best model is retrieved (Experiment P_avg_data_AutoML). The parameters are not thuned because the goal of the study is not to perfectly predict P_avg but to detect anomalies.
# MAGIC 
# MAGIC 
# MAGIC Then, the model is applied on data considered as outliers by the isolation forest method to evaluate the size of the energy production. 

# COMMAND ----------

data_P_avg = dfiforest1.to_pandas()
data_P_avg_out = data_P_avg[(data_P_avg.outlier_iforest==1) ]

# Load model we get with AutoML
model_name = "reg_P_avg"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
pred_P_avg = model.predict(data_P_avg_out)

# COMMAND ----------

# MAGIC %md
# MAGIC The difference between predicted and actual production is calculated. We consider the cases where the predicted production is higher than the actual production, because, in this case, there is a loss of profit for the company.

# COMMAND ----------

data_P_avg_out.loc[:, 'P_avg_pred'] = pred_P_avg
data_earning = data_P_avg_out[['Date_time', 'P_avg', 'P_avg_pred']]
data_earning.loc[:, 'Delta_P'] = data_earning.loc[:, 'P_avg_pred']- data_earning.loc[:, 'P_avg']
data_earning = data_earning[data_earning.loc[:, 'Delta_P'] >0 ]
print('Sum Average Production :', data_earning.Delta_P.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion

# COMMAND ----------

# MAGIC %md
# MAGIC At least 70 561 kW would have been produced by Engie if the wind turbines had not encountered any anomalies. Moreover, it is difficult to quantify the gains obtained by doing maintenance before the equipment breaks compared to the costs of breaking and stopping the wind turbine for repair.

# COMMAND ----------

# MAGIC %md
# MAGIC For a conventional train, we can therefore consider that it consumes between 60 and 100 Wh for 1 km traveled per passenger.
# MAGIC In a train of tgv we have 500 seats. So, if the train is full, it consumes 50 000 Wh per km. After calculation, with 70 561kW, a train can run for km. The gains of energy if there is no anomalies can power a Paris-Marseille round-trip.

# COMMAND ----------

# MAGIC %md
# MAGIC # Dashboard: Microsoft Power BI 
# MAGIC In addition to the work done previously, we have launched a Power BI to create an interface that allows you to easily view your data and share information within your organization. 
# MAGIC 
# MAGIC In order to implement this business analytics solution, we need to get the data from Azure. We used Partner connect to connect Power BI Desktop to our Azure Databricks clusters and Databricks SQL endpoints.
# MAGIC 
# MAGIC We linked the tables saved in Data directly to Power BI and prepared some flexible data visualizations using filters.

# COMMAND ----------

# MAGIC %md
# MAGIC <img width=45% src='https://raw.githubusercontent.com/Kroxyl/databricks-img/main/Pavg-year.png'>
# MAGIC <img width=45% src='https://raw.githubusercontent.com/Kroxyl/databricks-img/main/Pavg-day.png'>

# COMMAND ----------

# MAGIC %md
# MAGIC <img width=45% src='https://raw.githubusercontent.com/Kroxyl/databricks-img/main/pavg_wsinf10.png'>
# MAGIC 
# MAGIC <img width=45% src='https://raw.githubusercontent.com/Kroxyl/databricks-img/main/gb1t-gb2t-ot.png'>

# COMMAND ----------

# MAGIC %md
# MAGIC <img width=45% src='https://raw.githubusercontent.com/Kroxyl/databricks-img/main/all_temperature_features.png'>
# MAGIC <img width=45% src='https://raw.githubusercontent.com/Kroxyl/databricks-img/main/log-gb1t.png'>

# COMMAND ----------


