# Databricks notebook source
# MAGIC %run ./setup_config

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC # Media Analytics - Real-Time View

# COMMAND ----------

# MAGIC %md # ACTIVE_USERS
# MAGIC 
# MAGIC Metric calculated by aggregating unique users over a '2' minute sliding window

# COMMAND ----------

def update_active_users(row): 
  active_users =  {"USER_COUNT" : row.active_users}
  publish_message_sns(json.dumps(active_users), sns_topic_updates, active_users_type)

# COMMAND ----------

from pyspark.sql.functions import window,approx_count_distinct
from pyspark.sql.functions import col

active_users = getKinesisConsumer(kinesis_stream_name).selectExpr("lcase(CAST(data as STRING)) as jsonData")\
  .selectExpr("get_json_object(jsonData, '$.user_id') as user", \
          "get_json_object(jsonData, '$.metrictype') as type", \
          "CAST(get_json_object(jsonData, '$.timestamp') / 1000 as Timestamp) as ts") \
  .filter("type = 'stream'") \
  .withWatermark("ts", "5 seconds") \
  .groupBy(
    window("ts", "60 seconds", "30 seconds")
  ).agg(approx_count_distinct("user").alias("active_users"))

# COMMAND ----------

active_users.select(col('window.start').alias('start'), col('window.end').alias("end"), 'active_users') \
        .writeStream \
        .trigger(processingTime='30 seconds')  \
        .foreach(update_active_users) \
        .start()

# COMMAND ----------

# MAGIC %md #RECENT_VIEWS 
# MAGIC 
# MAGIC Metric calculated by aggregating unique video 'PLAY' event over a '2' minute sliding window

# COMMAND ----------

def update_video(row): 
  recent_views = {
              "VIDEOID" : row['video_id'],
              "VIEWS" : row['recent_views']
            }
  publish_message_sns(json.dumps(recent_views), sns_topic_updates, recent_views_type)

# COMMAND ----------

from pyspark.sql.functions import window,approx_count_distinct

video_views = getKinesisConsumer(kinesis_stream_name).selectExpr("lcase(CAST(data as STRING)) as jsonData")\
  .selectExpr("get_json_object(jsonData, '$.user_id') as user_id", \
          "get_json_object(jsonData, '$.metrictype') as type", \
          "get_json_object(jsonData, '$.video_id') as video_id", \
          "CAST(get_json_object(jsonData, '$.timestamp') / 1000 as Timestamp) as ts") \
  .filter("type = 'play'") \

# COMMAND ----------

recent_views = video_views \
  .withWatermark("ts", "5 seconds") \
  .groupBy(
    window("ts", "120 seconds", "30 seconds"),
    "video_id"
  ).agg(approx_count_distinct("user_id").alias("recent_views"))

# COMMAND ----------

recent_views.select('recent_views', 'video_id') \
           .writeStream \
           .foreach(update_video) \
           .start()

# COMMAND ----------

display(recent_views)

# COMMAND ----------

# MAGIC %md # Total Views Update

# COMMAND ----------

def update_total_views(batch, id):
  
    updates = batch.groupBy("video_id").count()
    
    for video in updates.collect():
        total_views = {
                      "VIDEOID" : video['video_id'],
                      "VIEWS" : video['count']
                      }
        publish_message_sns(json.dumps(total_views), sns_topic_updates, total_views_type)

# COMMAND ----------

getKinesisConsumer(kinesis_stream_name).selectExpr("lcase(CAST(data as STRING)) as jsonData")\
      .selectExpr("get_json_object(jsonData, '$.user_id') as user_id", \
              "get_json_object(jsonData, '$.metrictype') as type", \
              "get_json_object(jsonData, '$.video_id') as video_id", \
              "CAST(get_json_object(jsonData, '$.timestamp') / 1000 as Timestamp) as ts") \
      .filter("type = 'play'") \
      .writeStream \
      .trigger(processingTime='10 seconds') \
      .foreachBatch(update_total_views) \
      .start()

# COMMAND ----------


