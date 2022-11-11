# Databricks notebook source
demo_name = 'lynch_media'
s3_bucket = 'mediaqosdemo-logs-997819012307-us-west-2'

# COMMAND ----------

sns_topic_arn = 'arn:aws:sns:us-west-2:997819012307:mediaqosdemo__email_notification'
sns_topic_updates = 'arn:aws:sns:us-west-2:997819012307:mediaqosdemo__aggregations'

# COMMAND ----------

region = 'us-west-2'
cdn_logs_bucket = "s3://"+ s3_bucket +"/cdn_logs"

kinesis_stream_name = 'mediaqosdemo' + "-playerlogs-stream"

bronze_database = demo_name + "_bronze"
silver_database = demo_name + "_silver"
gold_database = demo_name + "_gold"

# COMMAND ----------

threshold = 0.0001

# COMMAND ----------

active_users_type = 'active_users'
recent_views_type = 'recent_views'
total_views_type = 'total_views'

# COMMAND ----------

import boto3, json

def invoke_lambda(event, function_name, region):
  
  lambda_client = boto3.client('lambda', region)
  
  lambda_client.invoke(
      FunctionName=function_name,
      InvocationType='Event',
      Payload=json.dumps(event),
    )

# COMMAND ----------

def getKinesisConsumer(stream_name): 
  return spark \
    .readStream \
    .format("kinesis") \
    .option("streamName", stream_name) \
    .option("initialPosition", "latest") \
    .option("region", region) \
    .load()

# COMMAND ----------

import boto3

def publish_message_sns(message, topic, update_type):

  sns_client = boto3.client('sns', region)

  response = sns_client.publish(
      TopicArn=topic,
      Message=message,
      Subject="updates",
      MessageStructure='string',
      MessageAttributes= {
          "update_type": {
             "DataType": "String", 
             "StringValue": update_type
          }
      })
