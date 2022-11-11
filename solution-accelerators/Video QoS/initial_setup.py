# Databricks notebook source
dbutils.widgets.get("demo")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC We assume that the data will be processed using tables as an abstraction.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS ${demo}_bronze CASCADE;
# MAGIC DROP DATABASE IF EXISTS ${demo}_silver CASCADE;
# MAGIC DROP DATABASE IF EXISTS ${demo}_gold CASCADE;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS ${demo}_bronze;
# MAGIC CREATE DATABASE IF NOT EXISTS ${demo}_silver;
# MAGIC CREATE DATABASE IF NOT EXISTS ${demo}_gold;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Bronze Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS ${demo}_bronze.cdn_logs;
# MAGIC 
# MAGIC CREATE TABLE ${demo}_bronze.cdn_logs (browserfamily string,
# MAGIC                   bytes string,
# MAGIC                   cdn_source string,
# MAGIC                   isbot boolean,
# MAGIC                   `location` string,
# MAGIC                   logdate date,
# MAGIC                   logtime string,
# MAGIC                   osfamily string,
# MAGIC                   requestid string, 
# MAGIC                   requestip string,
# MAGIC                   resulttype string,
# MAGIC                   year int,
# MAGIC                   month int,
# MAGIC                   day int, 
# MAGIC                   hour int)
# MAGIC USING json 
# MAGIC LOCATION 's3://$s3bucket/bronze/cdn_logs/'

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS ${demo}_bronze.player_events;
# MAGIC 
# MAGIC CREATE TABLE ${demo}_bronze.player_events (type string,
# MAGIC                   ts timestamp,
# MAGIC                   jsonData string)
# MAGIC USING delta
# MAGIC LOCATION 's3://$s3bucket/bronze/player_events/'

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from ${demo}_bronze.player_events;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Silver Tables

# COMMAND ----------

spark.table("andrei.products")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS ${demo}_silver.cdn_logs;
# MAGIC 
# MAGIC CREATE TABLE ${demo}_silver.cdn_logs (browserfamily string,
# MAGIC                   bytes string,
# MAGIC                   cdn_source string,
# MAGIC                   isbot boolean,
# MAGIC                   origin string,
# MAGIC                   `location` string,
# MAGIC                   logdate date,
# MAGIC                   logtime string,
# MAGIC                   osfamily string,
# MAGIC                   requestid string, 
# MAGIC                   ip string,
# MAGIC                   resulttype string,
# MAGIC                   year int,
# MAGIC                   month int,
# MAGIC                   day int, 
# MAGIC                   hour int)
# MAGIC USING delta 
# MAGIC LOCATION 's3://$s3bucket/silver/cdn_logs/'

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS ${demo}_silver.player_events;
# MAGIC 
# MAGIC CREATE TABLE ${demo}_silver.player_events (
# MAGIC                   user_id string,
# MAGIC                   type string,
# MAGIC                   video_id string,
# MAGIC                   avg_bitrate long,
# MAGIC                   connection_type string,
# MAGIC                   package string,
# MAGIC                   fps double,
# MAGIC                   cdn_tracking_id string,
# MAGIC                   duration long,
# MAGIC                   playlist_type string,
# MAGIC                   time_millisecond long,
# MAGIC                   `at` double,
# MAGIC                   buffer_type string, 
# MAGIC                   ts timestamp,
# MAGIC                   day int, 
# MAGIC                   hour int)
# MAGIC USING delta 
# MAGIC LOCATION 's3://$s3bucket/silver/player_events/'

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from ${demo}_silver.player_events;
