# Databricks notebook source
# MAGIC %run ./setup_config

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC # Sending email alerts using Structured Streaming and AWS SNS

# COMMAND ----------

import boto3

def invoke_sns_email(row):
  
  error_message = 'Number of errors for App has exceded the threshold {}'.format(row['percentage'])
  print(error_message)
  sns_client = boto3.client('sns', region)

  response = sns_client.publish(
      TopicArn=sns_topic_arn,
      Message=error_message,
      Subject="APP ERRORS",
      MessageStructure='string')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from mediaqosdemo_bronze.player_events

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

@pandas_udf("percentage float", PandasUDFType.GROUPED_MAP)
def calculate_error_percentage(pdf):

  grouped_pdf = pdf.groupby('type').count()
  total = len(pdf) 
  
  if 'stream' in grouped_pdf.index:
      df = pd.DataFrame([[grouped_pdf.at['stream', 'ts'] / total]],
                    columns=['percentage'])
  else:    
      df =  pd.DataFrame(columns=['percentage'])
      
  return df

# COMMAND ----------

(spark.readStream.table(bronze_database + '.player_events')\
    .groupBy()\
    .apply(calculate_error_percentage)\
    .where("percentage > {}".format(threshold)) \
    .writeStream
    .trigger(once=True) 
    .foreach(invoke_sns_email)
    .start())

# COMMAND ----------


