# Databricks notebook source
# DBTITLE 1,Configure locations of videos to be extracted and images to be saved
# Source Training Videos
srcVideoPath = "/databricks-datasets/cctvVideos/train/"

# Source Test Videos
srcTestVideoPath = "/databricks-datasets/cctvVideos/test/"

# Source Training Videos MP4
srcVideoMP4Path = "/databricks-datasets/cctvVideos/mp4/train/"

# Source Training Videos MP4
srcTestVideoMP4Path = "/databricks-datasets/cctvVideos/mp4/test/"

# CCTV Video Labeled Data
# TODO: This should be in /databricks-datasets/cctvVideos/labels/
labeledDataPath = "/mnt/tardis/videos/cctvFrames_train_labels/cctvFrames_train_labels.csv"


#
# Please configure the directory paths below
#

# Extracted Training Images (from Videos) Path
targetImgPath = "/mnt/tardis/videos/cctvFrames/train/"

# Extract Test Images (from Videos) Path
targetImgTestPath = "/mnt/tardis/videos/cctvFrames/test/"
#/mnt/raela/cctv_features_test/

# Extract Features from Images Path
imgFeaturesPath = "/mnt/tardis/videos/cctv_features/train/"

# Extract Test Features from Images Path
imgFeaturesTestPath = "/mnt/tardis/videos/cctv_features/test/"


# Print out values
print("Training Videos (srcVideoPath): %s" % srcVideoPath)
print("Test Videos (srcTestVideoPath): %s" % srcTestVideoPath)
print("Training Images (targetImgPath): %s" % targetImgPath)
print("Test Images (targetImgTestPath): %s" % targetImgTestPath)
print("Training Images Features (imgFeaturesPath): %s" % imgFeaturesPath)
print("Test Images Features (imgFeaturesTestPath): %s" % imgFeaturesTestPath)
print("Labeled Data (labeledDataPath): %s" % labeledDataPath)
print("Training MP4 Videos (srcVideoMP4Path): %s" % srcVideoMP4Path)
print("Test MP4 Videos (srcTestVideoMP4Path): %s" % srcTestVideoMP4Path)

# COMMAND ----------

# DBTITLE 1,display Images and Video Helper functions
# displayVid(): Shows video from mounted cloud storage
def displayVid(filepath):
  return displayHTML("""
  <video width="480" height="320" controls>
  <source src="/files/%s" type="video/mp4">
  </video>
  """ % filepath)

# displayDbfsVid(): Shows video from DBFS
def displayDbfsVid(filepath):
  return displayHTML("""
  <video width="480" height="320" controls>
  <source src="/dbfs/%s" type="video/mp4">
  </video>
  """ % filepath)

# displayImg(): Shows image from dbfs/cloud storage
def displayImg(filepath):
  dbutils.fs.cp(filepath, "FileStore/%s" % filepath)
  return displayHTML("""
  <img src="/files/%s">
  """ % filepath)
