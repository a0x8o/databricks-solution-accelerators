# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/vadim/dlt-demo-slides/main/images/e2e-ml.png" width="1000px" />

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance: model training
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged turbines.
# MAGIC 
# MAGIC A damaged inactive turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC Our data consist of vibration readings coming off sensors located in the gearboxes of gas turbines. 
# MAGIC 
# MAGIC 
# MAGIC We will be implementing:
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/wind_turbine/turbine-ds-flow-0.png" width="1000px" />
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right; margin: -10px 50px 0px 50px">
# MAGIC   <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width="400px" /><br/>
# MAGIC   *locations of the sensors*
# MAGIC </div>
# MAGIC 
# MAGIC We will use Gradient Boosted Tree Classification to predict which set of vibrations could be indicative of a failure.
# MAGIC 
# MAGIC One the model is trained, we'll use MFLow to track its performance and save it in the registry to deploy it in production

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all=$reset_all_data

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration
# MAGIC What do the distributions of sensor readings look like for our turbines? 
# MAGIC 
# MAGIC _Plot as bar charts using `summary` as Keys and all sensor Values_

# COMMAND ----------

_val_cols = ['AN3', 'AN4', 'AN5', 'AN6', 'AN7', 'AN8', 'AN9', 'AN10']

display(dataset.select(*_val_cols).summary())

# COMMAND ----------

# DBTITLE 1,Databricks ML Runtime Provides Common DS tools
import seaborn as sns

gold_turbine_dfp = dataset.sample(0.05, seed=314).toPandas()

valuable_cols = ['AN3', 'AN4', 'AN9' ,'status']

g = sns.PairGrid(gold_turbine_dfp[valuable_cols],
                 diag_sharey=True,
                 hue="status")

g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(sns.kdeplot)

# COMMAND ----------

# DBTITLE 1,Once the data are ready, train a model
#once the data is ready, we can train a model
import mlflow
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.mllib.evaluation import MulticlassMetrics

with mlflow.start_run() as mlrun:
  #the source table will automatically be logged to mlflow
  mlflow.spark.autolog()

  training, test = dataset.limit(1000).randomSplit([0.9, 0.1], seed = 5)
  
  gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
  grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3,4,5,10,15,25,30]).build()

  metrics = MulticlassClassificationEvaluator(metricName="f1")
  cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=metrics, numFolds=2)

  featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
  stages = [VectorAssembler(inputCols=featureCols, outputCol="va"),
            StandardScaler(inputCol="va", outputCol="features"),
            StringIndexer(inputCol="status", outputCol="label"), cv]
  
  pipeline = Pipeline(stages=stages)
  pipelineTrained = pipeline.fit(training)
  
  predictions = pipelineTrained.transform(test)
  metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd)
  
  # log key metrics
  mlflow.log_metric("precision", metrics.precision(1.0))
  mlflow.log_metric("accuracy", metrics.accuracy)
  mlflow.log_metric("f1", metrics.fMeasure(0.0, 2.0))
  
  # log the model
  mlflow.spark.log_model(pipelineTrained, artifact_path='turbine_gbt')
  mlflow.set_tag("model", "turbine_gbt")
  
  # add confusion matrix to the model
  labels = pipelineTrained.stages[2].labels
  fig = plt.figure()
  sns.heatmap(pd.DataFrame(metrics.confusionMatrix().toArray()), annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
  plt.suptitle("Turbine Damage Prediction. F1={:.2f}".format(metrics.fMeasure(1.0)), fontsize = 18)
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  mlflow.log_figure(fig, "confusion_matrix.png")

# COMMAND ----------

  mlflow.spark.log_model(pipelineTrained, artifact_path='turbine_gbt')

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC 
# MAGIC ## 3/ Saving our model to MLFLow registry
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/wind_turbine/turbine-ds-flow-3.png" width="700px" style="float: right" />
# MAGIC 
# MAGIC Our model is now fully packaged. MLflow tracked every steps, and logged the full model for us.
# MAGIC 
# MAGIC The next step is now to get the best run out of MLFlow and move it to our model registry. Your Data Engineering team will then be able to retrieve it and use it to run inferences at scale, or deploy it using REST api for real-time use cases.
# MAGIC 
# MAGIC *Note: this step is typically involving hyperparameter tuning. Databricks AutoML setup all that for you.*

# COMMAND ----------

# DBTITLE 1,Get the best model from the registry
best_model = mlflow.search_runs(filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED" and metrics.f1 > 0', max_results=1).iloc[0]
model_registered = mlflow.register_model("runs:/" + best_model.run_id + "/team5_turbine", "flight_school_gas_turbine_maintenance")

# COMMAND ----------

# DBTITLE 1,Flag version as staging/production ready
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
client.transition_model_version_stage(name = f"flight_school_gas_turbine_maintenance",
                                      version = model_registered.version,
                                      stage = "Production",
                                      archive_existing_versions = True)

# COMMAND ----------

# MAGIC %md #Deploying & using our model in production
# MAGIC 
# MAGIC Now that our model is in our MLFlow registry, we can start to use it in a production pipeline.

# COMMAND ----------

# MAGIC %md ### Scaling inferences using Spark 
# MAGIC We'll first see how it can be loaded as a spark UDF and called directly in a SQL function:

# COMMAND ----------

# DBTITLE 1,Let's load the model from the registry
get_turbine_status_udf = mlflow.pyfunc.spark_udf(spark, "models:/flight_school_gas_turbine_maintenance/Production", "string")
#Save the mdoel as SQL function (we could call it using python too)
spark.udf.register("get_turbine_status", get_turbine_status_udf)

# COMMAND ----------

# DBTITLE 1,Let's call the registered model in SQL!
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *, get_turbine_status(struct(AN3, AN4, AN5, AN6, AN7, AN8, AN9, AN10)) as status_forecast FROM turbine_gold_for_ml
