# Databricks notebook source
# DBTITLE 0,Video Identification of Suspicious Behavior: Preparation
# MAGIC %md 
# MAGIC 
# MAGIC # Video Identification of Suspicious Behavior: Preparation
# MAGIC 
# MAGIC This notebook will **prepare** your video data by:
# MAGIC * Processing the images by extracting out individual images (using `open-cv`) and saving them to DBFS / cloud storage
# MAGIC * With the saved images, execute Spark Deep Learning Pipelines `DeepImageFeaturizer` to extract image features and saving them to DBFS / cloud storage in Parquet format
# MAGIC  * Perform this task for both the training and test datasets
# MAGIC 
# MAGIC The source data used in this notebook can be found at [EC Funded CAVIAR project/IST 2001 37540](http://homepages.inf.ed.ac.uk/rbf/CAVIAR/)
# MAGIC 
# MAGIC <img src="/files/mnt/raela/video_splash.png" width=900/>
# MAGIC 
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

# DBTITLE 1,Directory of Training Videos
display(dbutils.fs.ls(srcVideoPath))

# COMMAND ----------

# DBTITLE 1,Example Video
# MAGIC %md
# MAGIC Here is an example video that we will perform our training on to identify suspicious behavior.
# MAGIC * The source of this data is from the [EC Funded CAVIAR project/IST 2001 37540](http://homepages.inf.ed.ac.uk/rbf/CAVIAR/).
# MAGIC 
# MAGIC  

# COMMAND ----------

displayDbfsVid("databricks-datasets/cctvVideos/mp4/train/Browse2.mp4")

# COMMAND ----------

# DBTITLE 1,Process Videos - Extract Video Frames
# MAGIC %md
# MAGIC Extract JPG images from MPG videos using OpenCV (`cv2`)

# COMMAND ----------

# Extract and Save Images using CV2
def extractImagesSave(src, tgt):
  import cv2
  import uuid
  import re

  ## Extract one video frame per second and save frame as JPG
  def extractImages(pathIn):
      count = 0
      srcVideos = "/dbfs" + src + "(.*).mpg"
      p = re.compile(srcVideos)
      vidName = str(p.search(pathIn).group(1))
      vidcap = cv2.VideoCapture(pathIn)
      success,image = vidcap.read()
      success = True
      while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite("/dbfs" + tgt + vidName + "frame%04d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1
        print ('Wrote a new frame')    

  ## Extract frames from all videos and save in s3 folder
  def createFUSEpaths(dbfsFilePath):
    return "/dbfs/" + dbfsFilePath[0][6:]
  
  # Build up fileList RDD
  fileList = dbutils.fs.ls(src)
  FUSEfileList = map(createFUSEpaths, fileList)
  FUSEfileList_rdd = sc.parallelize(FUSEfileList)
  
  # Ensure directory is created
  dbutils.fs.mkdirs(tgt)
  
  # Extract and save images
  FUSEfileList_rdd.map(extractImages).count()

# COMMAND ----------

# MAGIC %md # Training Dataset

# COMMAND ----------

# DBTITLE 1,Extract Training Images
# Extract Images
extractImagesSave(srcVideoPath, targetImgPath)

# View file list of images extracted from video
display(dbutils.fs.ls(targetImgPath))

# COMMAND ----------

# DBTITLE 1,Review Training Images
from pyspark.ml.image import ImageSchema

trainImages = ImageSchema.readImages(targetImgPath)
display(trainImages)

# COMMAND ----------

# DBTITLE 1,Feature Extraction using DeepImageFeaturizer
# MAGIC %md
# MAGIC Use [Spark Deep Learning Pipelines](https://github.com/databricks/spark-deep-learning) `DeepImageFeaturizer` to build image features via the InceptionV3 model

# COMMAND ----------

# DBTITLE 0,Save Features Function
# Save Image Features using 
def saveImageFeatures(images, filePath):
  from sparkdl import DeepImageFeaturizer

  # Build featurizer using DeepImageFeaturizer and the InceptionV3 model
  featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")

  # Transform images to pull out image (origin, height, width, nChannels, mode, data) and features (udt)
  features = featurizer.transform(images)

  # Push feature information into Parquet file format
  # This might take a few minutes
  dbutils.fs.mkdirs(filePath)

  # Extract only image file name (imgFileName) within our saved features
  features.select("image.origin", "features").coalesce(2).write.mode("overwrite").parquet(filePath)

# COMMAND ----------

# DBTITLE 1,Save Training Image Features
saveImageFeatures(trainImages, imgFeaturesPath)

# COMMAND ----------

# View Parquet Features
display(dbutils.fs.ls(imgFeaturesPath))

# COMMAND ----------

# MAGIC %md # Test Dataset

# COMMAND ----------

# DBTITLE 1,Directory of Test Videos
display(dbutils.fs.ls(srcTestVideoPath))

# COMMAND ----------

# DBTITLE 1,Extract Test Images
# Extract Images
extractImagesSave(srcTestVideoPath, targetImgTestPath)

# View file list of images extracted from video
display(dbutils.fs.ls(targetImgTestPath))

# COMMAND ----------

# DBTITLE 1,Review Test Images
from pyspark.ml.image import ImageSchema

testImages = ImageSchema.readImages(targetImgTestPath)
display(testImages)

# COMMAND ----------

# DBTITLE 1,Save Test Image Features
saveImageFeatures(testImages, imgFeaturesTestPath)

# COMMAND ----------

# View Parquet Features
display(dbutils.fs.ls(imgFeaturesTestPath))
