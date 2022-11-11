# Databricks notebook source
# DBTITLE 0,Video Identification of Suspicious Behavior: Preparation
# MAGIC %md 
# MAGIC 
# MAGIC # Video Identification of Suspicious Behavior
# MAGIC 
# MAGIC This notebook will process your video data by:
# MAGIC * Utilize the data processed in the `Video Identification of Suspicious Behavior: Preparation`
# MAGIC * Load the training data
# MAGIC * Train the model against the training data
# MAGIC * Generate predictions against the test data using this model
# MAGIC * Any suspicious activity in our videos?
# MAGIC 
# MAGIC The source data used in this notebook can be found at [EC Funded CAVIAR project/IST 2001 37540](http://homepages.inf.ed.ac.uk/rbf/CAVIAR/)
# MAGIC 
# MAGIC <img src="/files/mnt/raela/video_splash.png" width=900/>
# MAGIC 
# MAGIC ### Prerequisite
# MAGIC * Execute the `Video Identification of Suspicious Behavior: Preparation` to setup the images and feature datasets
# MAGIC 
# MAGIC ### Cluster Configuration
# MAGIC * Suggested cluster configuration:
# MAGIC  * Databricks Runtime Version: `Databricks Runtime for ML` (e.g. 4.1 ML, 4.2 ML, etc.)
# MAGIC  * Driver: `r4.4xlarge`
# MAGIC  * Workers: `r4.4xlarge (2)`
# MAGIC  * Python: `Python 3`
# MAGIC  
# MAGIC ### Need to install manually
# MAGIC To install, refer to **Upload a Python PyPI package or Python Egg** [Databricks](https://docs.databricks.com/user-guide/libraries.html#upload-a-python-pypi-package-or-python-egg) | [Azure Databricks](https://docs.azuredatabricks.net/user-guide/libraries.html#upload-a-python-pypi-package-or-python-egg)
# MAGIC 
# MAGIC * Python Libraries:
# MAGIC  * `opencv-python`: 3.4.2 
# MAGIC  
# MAGIC ### Libraries Already Included in Databricks Runtime for ML
# MAGIC Because we're using *Databricks Runtime for ML*, you do **not** need to install the following libraires
# MAGIC * Python Libraries:
# MAGIC  * `h5py`: 2.7.1
# MAGIC  * `tensorflow`: 1.7.1
# MAGIC  * `keras`: 2.1.5 (Using TensorFlow backend)
# MAGIC  * *You can check by `import tensorflow as tf; print(tf.__version__)`*
# MAGIC 
# MAGIC * JARs:
# MAGIC  * `spark-deep-learning-1.0.0-spark2.3-s_2.11.jar`
# MAGIC  * `tensorframes-0.3.0-s_2.11.jar`
# MAGIC  * *You can check by reviewing cluster's Spark UI > Environment)*

# COMMAND ----------

# DBTITLE 1,Include Video Configuration and Display Helper Functions
# MAGIC %run ./video_config

# COMMAND ----------

# DBTITLE 1,Load Training Data
# MAGIC %md 
# MAGIC * Read the Parquet files previously generated containing the training dataset
# MAGIC * Read the hand labelled data 

# COMMAND ----------

# Prefix to add prior to join
prefix = "dbfs:" + targetImgPath

# Read in hand-labeled data 
from pyspark.sql.functions import expr
labels = spark.read.csv(labeledDataPath, header=True, inferSchema=True)
labels_df = labels.withColumn("filePath", expr("concat('" + prefix + "', ImageName)")).drop('ImageName')

# Read in features data (saved in Parquet format)
featureDF = spark.read.parquet(imgFeaturesPath)

# Create train-ing dataset by joining labels and features
train = featureDF.join(labels_df, featureDF.origin == labels_df.filePath).select("features", "label", featureDF.origin)

# Validate number of images used for training
train.count()

# COMMAND ----------

# DBTITLE 1,Train our Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

# Fit LogisticRegression Model
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
lrModel = lr.fit(train)

# COMMAND ----------

# DBTITLE 1,Generate Predictions on Test data
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

# Load Test Data
featuresTestDF = spark.read.parquet(imgFeaturesTestPath)

# Generate predictions on test data
result = lrModel.transform(featuresTestDF)
result.createOrReplaceTempView("result")

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Extract first and second elements of the StructType
firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())

# Second element is what we need for probability
predictions = result.withColumn("prob2", secondelement('probability'))
predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# DBTITLE 1,Any suspicious activity in our videos?
# MAGIC %sql
# MAGIC select origin, probability, prob2, prediction from predictions where prediction = 1  order by prob2 desc

# COMMAND ----------

# DBTITLE 1,View Top 3 frames
# MAGIC %md 
# MAGIC View the top three most suspicious images based on `prob2` column

# COMMAND ----------

displayImg("dbfs:/mnt/tardis/videos/cctvFrames/test/Fight_OneManDownframe0024.jpg")

# COMMAND ----------

displayImg("dbfs:/mnt/tardis/videos/cctvFrames/test/Fight_OneManDownframe0014.jpg")

# COMMAND ----------

displayImg("dbfs:/mnt/tardis/videos/cctvFrames/test/Fight_OneManDownframe0017.jpg")

# COMMAND ----------

# MAGIC %md ## View the Source Video
# MAGIC View the source video of the suspicious images

# COMMAND ----------

displayDbfsVid("databricks-datasets/cctvVideos/mp4/test/Fight_OneManDown.mp4")
