# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Quality of Service Demo
# MAGIC 
# MAGIC <img src="https://blog.solidsignal.com/wp-content/uploads/2018/02/super-fast-gigabit-interne_tEHzBNx.jpg">
# MAGIC 
# MAGIC ## Instructions
# MAGIC 1. Make sure that instance profile `databricks-access-s3` is attached to the cluster
# MAGIC 1. Attach `iso3166` from Pypi into cluster
# MAGIC 1. Open the [streaming service](https://dx9a2eqns22gn.cloudfront.net/ui/index.html).
# MAGIC 1. Click on a "sample video"
# MAGIC 1. Click "Play"
# MAGIC 1. Click "Pause"
# MAGIC 1. Open [01. Data Preparation](https://field-eng.cloud.databricks.com/#notebook/2418106/command/2418124) notebook and display streaming logs
# MAGIC 
# MAGIC ## Talk track
# MAGIC 1. We start with streaming player events from kinesis into a bronze delta table (append-only).
# MAGIC 1. We then proceed to stream 
# MAGIC 1. We also stream CDN JSON logs from an S3 bucket directly into a silver delta table using auto loader.
# MAGIC 1. ...
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Resources:
# MAGIC - Blog: https://databricks.com/blog/2020/05/06/how-to-build-a-quality-of-service-qos-analytics-solution-for-streaming-video-services.html
# MAGIC - Demo Video: https://www.youtube.com/watch?v=Q_LEdPH3rTw&list=PLTPXxbhUt-YWIG6Eos76FGnk8077OibtQ&index=3&t=0s
# MAGIC - QoS Workshop Slides: https://docs.google.com/presentation/d/1MPdlof2d2WFKSdfvsBUgdvt3XKDbpHhHfLf0ZePIbW8/edit?ts=5eb2f5af#slide=id.g6e69bad7a4_1_2955
# MAGIC 
# MAGIC AWS Services:
# MAGIC - Webservice: https://dx9a2eqns22gn.cloudfront.net/ui/index.html
# MAGIC - Cloudfront: https://console.aws.amazon.com/cloudfront/home?region=us-west-2#distribution-settings:E2GMMQ3RNA5EN8
# MAGIC - AppSync: https://us-west-2.console.aws.amazon.com/appsync/home?region=us-west-2#/r472rjetczalfeyc64jc2g5dli/v1/home
# MAGIC - AppSync source S3 bucket: https://s3.console.aws.amazon.com/s3/buckets/mediaqosdemo-sourcebucket-1nknixcja7wvz?region=us-west-2&tab=objects
