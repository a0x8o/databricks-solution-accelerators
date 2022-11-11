# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC #Data Ingestion

# COMMAND ----------

# MAGIC %run ./setup_config

# COMMAND ----------

# MAGIC %md #Ingestion -> Bronze Tables

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Player Apps Events from Kinesis are pushed directly to a Delta append-only table. 

# COMMAND ----------

kinesis_df = spark.readStream \
            .format("kinesis") \
            .option("streamName", kinesis_stream_name) \
            .option("initialPosition", "latest") \
            .option("region", region) \
            .load()

transformed_df = kinesis_df.selectExpr("lcase(CAST(data as STRING)) as jsonData")\
                .selectExpr("get_json_object(jsonData, '$.metrictype') as type", \
                     "CAST(get_json_object(jsonData, '$.timestamp') / 1000 as Timestamp) as ts",
                      "jsonData")

transformed_df.writeStream\
            .trigger(processingTime='20 seconds')\
            .format("delta")\
            .outputMode("append")\
            .option('checkpointLocation', 's3://{}/checkpoint/{}/player_events'.format(s3_bucket, bronze_database))\
            .table(bronze_database + ".player_events")

# COMMAND ----------

display(transformed_df)

# COMMAND ----------

# MAGIC %md #Bronze -> Silver

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Player Events 
# MAGIC 
# MAGIC #### Bronze Delta Table -> Structured Streaming -> Silver Delta Table

# COMMAND ----------

player_events_df = spark.readStream \
                  .format('delta') \
                  .table(bronze_database + '.player_events')      

# COMMAND ----------

display(player_events_df)

# COMMAND ----------

from pyspark.sql.functions import get_json_object, col, dayofmonth, hour
from pyspark.sql.types import StringType

player_silver_df = player_events_df.select(
       'type', 
        get_json_object(col('jsonData'), '$.user_id').alias('user_id').cast('string'),
        get_json_object(col('jsonData'), '$.video_id').alias('video_id').cast('string'),
        get_json_object(col('jsonData'), '$.avg_bitrate').alias('avg_bitrate').cast('long'),
        get_json_object(col('jsonData'), '$.connection_type').alias('connection_type').cast('string'),
        get_json_object(col('jsonData'), '$.package').alias('package'),
        get_json_object(col('jsonData'), '$.fps').alias('fps').cast('double'),
        get_json_object(col('jsonData'), '$.cdn_tracking_id').alias('cdn_tracking_id'),  
        get_json_object(col('jsonData'), '$.duration').alias('duration').cast('long'),
        get_json_object(col('jsonData'), '$.playlist_type').alias('playlist_type'),
        get_json_object(col('jsonData'), '$.time_millisecond').alias('time_millisecond').cast('long'),
        get_json_object(col('jsonData'), '$.at').alias('at').cast('double'),
        get_json_object(col('jsonData'), '$.buffer_type').alias('buffer_type'),
        'ts',
        dayofmonth(col('ts')).alias('day'),
        hour(col('ts')).alias('hour'))

# COMMAND ----------

player_silver_df.writeStream\
        .trigger(processingTime='60 seconds') \
        .option('checkpointLocation',  's3://{}/checkpoint/{}/silver/player_events'.format(s3_bucket, silver_database))\
        .format('delta') \
        .table(silver_database + '.player_events')

# COMMAND ----------

# MAGIC %md 
# MAGIC ### CDN LOGS
# MAGIC 
# MAGIC ####Use Auto Loader to load JSON -> Silver CDN Delta using Databricks Auto-Loader & IP Anonymization

# COMMAND ----------

from pyspark.sql.functions import col, udf
import ipaddress
import re
import requests

#create a UDF to parse IPs to country
@udf("string")
def map_ip_to_location(row):
  headers = {"Authorization": "Bearer ---"}
  r = requests.get(url = "https://ipinfo.io/" + str(row), headers = headers)
  print(r.json())
  try :
     country = countries.get(r.json()['country']).alpha3
  except :
     country = 'Unavailable'
  return country

#create a UDF to anonymize IPs
@udf("string")
def ip_anonymizer(value):
  ip = ipaddress.ip_address(value)
  if ip.version == 4: 
    return re.sub(r'\.\d*$','.0',value)
  elif ip.version == 6: 
    return re.sub(r'[\da-f]*:[\da-f]*$', "0000:0000",value)
  return 'unknown'  

# COMMAND ----------

from pyspark.sql.functions import lit

json_schema = table(bronze_database + ".cdn_logs").schema

auto_loader_df = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.format", "json") \
  .option("cloudFiles.region", region) \
  .schema(json_schema) \
  .load(cdn_logs_bucket)

anonymized_df = auto_loader_df.select('*', ip_anonymizer('requestip').alias('ip'))\
          .drop('requestip')\
          .withColumn("origin", map_ip_to_location("ip"))

#Trigger.once()
anonymized_df.writeStream \
          .trigger(once=True) \
          .option('checkpointLocation', 's3://{}/checkpoint/{}/cdn_logs'.format(s3_bucket, silver_database))\
          .format('delta') \
          .table(silver_database + '.cdn_logs')

# COMMAND ----------

display(table(silver_database + '.cdn_logs'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Silver --> Gold
# MAGIC 
# MAGIC Join CDN logs and player events to a gold table.
# MAGIC 
# MAGIC Data to be refreshed on demand / scheduled job.

# COMMAND ----------

from pyspark.sql.functions import lower, col

player_logs = table('{}.player_events'.format(silver_database))
cdn_logs = table('{}.cdn_logs'.format(silver_database))

df = cdn_logs.drop("hour").drop("day")\
           .withColumn('cdn_tracking_id', lower(col('requestid'))) \
           .join(player_logs, 'cdn_tracking_id')

df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(gold_database + ".video_activity")

# COMMAND ----------

display(table('{}.video_activity'.format(gold_database)))

# COMMAND ----------


