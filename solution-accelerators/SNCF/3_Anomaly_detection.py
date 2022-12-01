# Databricks notebook source
# MAGIC %md 
# MAGIC # Challenge Databricks x Microsoft: Wind Turbine Anomaly Detection
# MAGIC # Part 3 : Modeling
# MAGIC 
# MAGIC **Authors**
# MAGIC - Amine HADJ-YOUCEF
# MAGIC - Maxime  CONVERT
# MAGIC - Cassandre DIAINE
# MAGIC - Axel DIDIER

# COMMAND ----------

import os
import time
from collections import Counter

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import pyspark.pandas as pd_sp
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd_sp.set_option('compute.ops_on_diff_frames', True)


# COMMAND ----------

df1s = spark.read.table('hackaton_team_sncf.df_post_turbine_R80711').to_pandas_on_spark().dropna()
df2s = spark.read.table('hackaton_team_sncf.df_post_turbine_R80721').to_pandas_on_spark().dropna()
df3s = spark.read.table('hackaton_team_sncf.df_post_turbine_R80736').to_pandas_on_spark().dropna()
df4s = spark.read.table('hackaton_team_sncf.df_post_turbine_R80790').to_pandas_on_spark().dropna()
# df_concat = spark.read.table('hackaton_team_sncf.df_post_turbine_all').to_pandas_on_spark()


# COMMAND ----------

turbine_names = ["R80711","R80721", "R80736", "R80790"]
df_turbine = {"R80711": df1s, "R80721": df2s, "R80736": df3s, "R80790":df4s} 


# COMMAND ----------

print("Dimension des datasets : \n", ["{} - {}".format(turbine, df_turbine[turbine].shape) for turbine in turbine_names])

# COMMAND ----------

# MAGIC %md # Modeling: Anomaly detection

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Classical approach

# COMMAND ----------

# MAGIC %md
# MAGIC An AutoML model was run to predict P_avg based on the avg column of the other sensors. The idea is to be able to retrieve the data exploration notebook. On this notebook, we find the distribution of the variables. For this classical approach, we separate the variables that have a Gaussian distribution from the other variables. For this outlier detection via thresholding, we are only interested in the columns of means xxx_avg of the variables with a Gaussian distribution.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gaussian distribution
# MAGIC 
# MAGIC For Gaussian distributions, outlier detection is done by thresholding according to quartiles. The variables with a Gaussian distribution are the following: Rt_avg, Q_avg, DB1t, DB2t, Dst, GB1t, GB2t, Git, Gost, Yt, Ws1, Ws2, Ot.

# COMMAND ----------

def get_low_up_lim(data_pdsp, col):
  ''' Consider as outliers values outside [Q1 - 1.5 * IQR , Q3 + 1.5 * IQR ]
  Return low_lim = Q1 - 1.5 * IQR  and up_lim = Q3 + 1.5 * IQR
  '''
  Q1 = data_pdsp.loc[:, col].quantile(0.25)
  Q3 = data_pdsp.loc[:, col].quantile(0.75)
  IQR = Q3 - Q1  
  low_lim = Q1 - 1.5 * IQR 
  up_lim = Q3 + 1.5 * IQR 
  print(col, "low_lim : ", low_lim, 'Q1 : ', Q1, "up_lim : ", up_lim, 'Q3 : ', Q3)

  return(low_lim, up_lim)

# COMMAND ----------

def get_classic_outlier(data):

  col_gaussian = ['Rt_avg', 'Q_avg', 'Dst_avg', 'Git_avg',
                  'Yt_avg', 'Ws1_avg', 'Ws2_avg', 'Ot_avg',
                  'Db1t_avg', 'Db2t_avg', 'Gb1t_avg', 'Gb2t_avg']

  for col in col_gaussian : 
    low_lim, up_lim = get_low_up_lim(data, col)
    colname_new = col + '_pb'
    data.loc[:, colname_new] = data.loc[:, col].apply(lambda x: 1 if (x<low_lim  or x>up_lim ) else 0)
  
  data_out_pd = data.to_pandas()  

  col_list_pb = [col for col in data_out_pd.columns if col.endswith('_pb')]
  data_out_pd.loc[:, "sum_outlier"] =  data_out_pd[col_list_pb].sum(axis = 1)
  data_out_pd.loc[:, "outlier"] =  data_out_pd.sum_outlier.apply(lambda x: 1 if x>0 else 0)
  
  data_out_sp=spark.createDataFrame(data_out_pd) 

  data_out_sp.write.format('delta').mode('overwrite').saveAsTable('hackaton_team_sncf.df_res_class_'+turbine)
  
  return(print("Done"))



# COMMAND ----------

for turbine in turbine_names: # Boucle for sur les datasets
    toc = 0
    tic = time.time()
    print("Données de la turbine \n", turbine)
    dfs = df_turbine[turbine]
    
    get_classic_outlier(dfs)

# COMMAND ----------

# MAGIC %md ## Unsupervised learning approaches

# COMMAND ----------

# MAGIC %md
# MAGIC We tested one clustering method (K-Means) as a starting point to play with the data, and further tested two other commonly used anomaly detection methods, namely Local Outlier Factor and Isolation Forest.

# COMMAND ----------


def plot_clusters(features_std, clusterer, cluster_labels, 
                  save_to="/dbfs/FileStore/shared_uploads/pahy04211@commun.ad.sncf.fr/", 
                  filename="", silhouette_avg=0):
  """
  This function plot a scatter plot using matplotlib 
  """
  fig = plt.figure()
  colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
  plt.scatter(
      features_std[:, 0], features_std[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
  )

  # Labeling the clusters
  centers = clusterer.cluster_centers_
  
  # Draw white circles at cluster centers
  plt.scatter(
      centers[:, 0],
      centers[:, 1],
      marker="o",
      c="white",
      alpha=1,
      s=200,
      edgecolor="k",
  )

  for i, c in enumerate(centers):
      plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

  plt.title(f"The visualization of the clustered data: {silhouette_avg}")
  plt.xlabel("Feature space for the 1st feature")
  plt.ylabel("Feature space for the 2nd feature")
  plt.show()
  fig.savefig(save_to + filename + ".png")
  return fig


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1 - Kmeans

# COMMAND ----------

# MAGIC %md 
# MAGIC #### using all features

# COMMAND ----------

from sklearn.metrics import silhouette_samples, silhouette_score

for turbine in turbine_names: # Boucle for sur les datasets
    toc = 0
    tic = time.time()
    print("Données de la turbine \n", turbine)
    dfs = df_turbine[turbine]
    
    # Selection des données
    dfs_np = dfs.loc[:, dfs.columns[1:]].to_numpy()
    n_rows, n_cols = dfs_np.shape
    
    # Clustering
    range_n_clusters = [2, 3]
    scaler = StandardScaler() # Normalisation des données: centré et réduit
    features_std = scaler.fit_transform(dfs_np)
    random_state = 0
    for n_clusters in range_n_clusters: # Boucle for sur les n clusters
    
      with mlflow.start_run(run_name=turbine + '_kmeans') as new_run:
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = clusterer.fit_predict(features_std)
        silhouette_avg = silhouette_score(features_std, cluster_labels)
        toc = time.time()
        
        print(
            "n_clusters =", n_clusters,
            "Average silhouette_score  :", silhouette_avg,
            "Temps =", toc-tic)
        
        plot_clusters(features_std, clusterer, cluster_labels, save_to="/dbfs/mnt/Input/Amine/Datathon/", filename=f'{turbine}_kmeans_{n_clusters}' )
        
        ##############
        ### MLFLOW ###
        ##############
        mlflow.log_param("turbine_name", turbine)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_cols", n_cols)
        mlflow.log_metric("time", toc-tic)
        mlflow.log_metric("silhouette_avg", silhouette_avg)
        mlflow.sklearn.autolog()





# COMMAND ----------

# MAGIC %md 
# MAGIC #### using reduced feature (by PCA)

# COMMAND ----------

from sklearn.metrics import silhouette_samples, silhouette_score

for turbine in turbine_names: # Boucle for sur les datasets
    toc = 0
    tic = time.time()
    print("Données de la turbine \n", turbine)
    dfs = df_turbine[turbine]
    
    # Selection des données
    dfs_np = dfs.loc[:, dfs.columns[1:]].to_numpy()
    n_rows, n_cols = dfs_np.shape

    # Réduction des dimensions ACP
    n_components = 2
    random_state=0
    pca = decomposition.PCA(n_components=n_components, random_state=random_state)
    dfs_comp = pca.fit_transform(dfs_np)
    print('Variance Expliquée =', pca.explained_variance_ratio_)
    
    # Clustering
    range_n_clusters = [2, 3]
    scaler = StandardScaler() # Normalisation des données: centré et réduit
    features_std = scaler.fit_transform(dfs_comp)

    for n_clusters in range_n_clusters: # Boucle for sur les n clusters
    
      with mlflow.start_run(run_name=turbine + '_kmeans_pca') as new_run:
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = clusterer.fit_predict(features_std)
        silhouette_avg = silhouette_score(features_std, cluster_labels)
        toc = time.time()
        
        print(
            "n_clusters =", n_clusters,
            "Average silhouette_score  :", silhouette_avg,
            "Temps =", toc-tic)
        
        plot_clusters(features_std, clusterer, cluster_labels, save_to="/dbfs/mnt/Input/Amine/Datathon/", filename=f'{turbine}_kmeans_{n_components}_{n_clusters}' )
        
        ##############
        ### MLFLOW ###
        ##############
        mlflow.log_param("turbine_name", turbine)
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_cols", n_cols)
        mlflow.log_metric("time", toc-tic)
        mlflow.log_metric("silhouette_avg", silhouette_avg)
        mlflow.sklearn.autolog()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 - Local Outlier Factor

# COMMAND ----------

from sklearn.neighbors import LocalOutlierFactor

outliers_fraction = 0.1

for turbine in turbine_names:
    toc = 0
    tic = time.time()
    print("Données de la turbine \n", turbine)
    dfs = df_turbine[turbine]
    
    # Selection des données
    dfs_np = dfs.loc[:,dfs.columns[1:]].to_numpy()#.drop('cluster')
    n_rows, n_cols = dfs_np.shape
  
    scaler = StandardScaler() # Normalisation des données: centré et réduit
    features_std = scaler.fit_transform(dfs_np)
      
    with mlflow.start_run(run_name=turbine+'_lof') as new_run:
      n_neighbors = 20
      lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=outliers_fraction)
      y_pred = lof.fit_predict(features_std)
      N_outlier = np.sum(y_pred==-1)
      
      dfs['outlier_lof'] = pd_sp.Series(y_pred, index=dfs.index.tolist())
      dfs.to_spark().write.format('delta').mode('overwrite').saveAsTable('hackaton_team_sncf.df_res_lof_'+turbine)
      
      toc = time.time()
      print(
          "For n_neighbors =", n_neighbors,
          "Time =", toc-tic,
          "N outlier =", N_outlier)

      ##############
      ### MLFLOW ###
      ##############
      mlflow.log_param("turbine_name", turbine)
      mlflow.log_param("n_rows", df1s.shape[0])
      mlflow.log_param("n_cols", df1s.shape[0])
      mlflow.log_param("n_outlier", N_outlier)
      mlflow.log_metric("time", toc-tic)
      mlflow.log_metric("n_neighbors", n_neighbors)
      mlflow.sklearn.autolog()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 - Isolation Forest

# COMMAND ----------

from sklearn.ensemble import IsolationForest

outliers_fraction = 0.1

for turbine in turbine_names:
    toc = 0
    tic = time.time()
    print("Données de la turbine \n", turbine)
    dfs = df_turbine[turbine]
    
    # Selection des données
    dfs_np = dfs.loc[:,dfs.columns[1:]].to_numpy()#.drop('cluster')
    n_rows, n_cols = dfs_np.shape
  
    scaler = StandardScaler() # Normalisation des données: centré et réduit
    features_std = scaler.fit_transform(dfs_np)
      
    with mlflow.start_run(run_name=turbine+'_iforest') as new_run:
      
      iforest = IsolationForest(contamination=outliers_fraction, random_state=0)

      y_pred = iforest.fit_predict(features_std)
      N_outlier = np.sum(y_pred==-1)
      
      dfs['outlier_iforest'] = pd_sp.Series(y_pred, index=dfs.index.tolist())
      dfs.to_spark().write.format('delta').mode('overwrite').saveAsTable('hackaton_team_sncf.df_res_iforest_'+turbine)
      
      toc = time.time()
      print(
          "For n_neighbors =", n_neighbors,
          "Time =", toc-tic,
          "N outlier =", N_outlier)

      ##############
      ### MLFLOW ###
      ##############
      mlflow.log_param("turbine_name", turbine)
      mlflow.log_param("n_rows", df1s.shape[0])
      mlflow.log_param("n_cols", df1s.shape[0])
      mlflow.log_param("n_outlier", N_outlier)
      mlflow.log_metric("time", toc-tic)
      mlflow.sklearn.autolog()

