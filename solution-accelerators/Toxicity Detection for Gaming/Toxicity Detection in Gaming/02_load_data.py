# Databricks notebook source
# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC ### In this notebook you:
# MAGIC * Create a database for the tables to reside in.
# MAGIC * Move the data downloaded in the notebook `01_intro` into object storage.
# MAGIC * Write the data out in `Delta` format.
# MAGIC * Create tables for easy access and querability.
# MAGIC * Explore the dataset and relationships.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create the database

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS gaming CASCADE;
# MAGIC CREATE DATABASE IF NOT EXISTS gaming;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Move training Data
# MAGIC 
# MAGIC Move the Jigsaw train and test data from the driver node to object storage so that it can be be ingested into Delta Lake.

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/train.csv", "dbfs:/tmp/train.csv")
dbutils.fs.mv("file:/databricks/driver/test.csv", "dbfs:/tmp/test.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Write Data to Delta Lake
# MAGIC 
# MAGIC In this section of the solution accelerator, we begin using [Delta Lake](https://delta.io/). 
# MAGIC * Delta Lake is an open-source project that enables building a **Lakehouse architecture** on top of existing storage systems such as S3, ADLS, GCS, and HDFS.
# MAGIC    * Information on the **Lakehouse Architecture** can be found in this [paper](http://cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf) that was presented at [CIDR 2021](http://cidrdb.org/cidr2021/index.html) and in this [video](https://www.youtube.com/watch?v=RU2dXoVU8hY)
# MAGIC 
# MAGIC * Key features of Delta Lake include:
# MAGIC   * **ACID Transactions**: Ensures data integrity and read consistency with complex, concurrent data pipelines.
# MAGIC   * **Unified Batch and Streaming Source and Sink**: A table in Delta Lake is both a batch table, as well as a streaming source and sink. Streaming data ingest, batch historic backfill, and interactive queries all just work out of the box. 
# MAGIC   * **Schema Enforcement and Evolution**: Ensures data cleanliness by blocking writes with unexpected.
# MAGIC   * **Time Travel**: Query previous versions of the table by time or version number.
# MAGIC   * **Deletes and upserts**: Supports deleting and upserting into tables with programmatic APIs.
# MAGIC   * **Open Format**: Stored as Parquet format in blob storage.
# MAGIC   * **Audit History**: History of all the operations that happened in the table.
# MAGIC   * **Scalable Metadata management**: Able to handle millions of files are scaling the metadata operations with Spark.
# MAGIC   
# MAGIC   
# MAGIC <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/delta-lake-raw.png"; width="50%" />

# COMMAND ----------

trainDF = spark.read.csv('dbfs:/tmp/train.csv',header=True,escape='"',multiLine=True)
trainDF.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("gaming.toxicity_training")

testDF = spark.read.csv('dbfs:/tmp/test.csv',header=True,escape='"',multiLine=True)
testDF.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("gaming.toxicity_test")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.1: Ingest Game Data into Delta Lake
# MAGIC 
# MAGIC Loop over the Dota 2 data files to:
# MAGIC * Move files to object storage
# MAGIC * Load data into Delta tables for analysis

# COMMAND ----------

for file in ['match','match_outcomes','player_ratings','players','chat','cluster_regions']:
  df = spark.read.csv(f"dbfs:/tmp/{file}.csv",header=True,escape='"',multiLine=True)
  df.write.format("delta").saveAsTable(f"Gaming.toxicity_{file}", overwrite=False)
  dbutils.fs.mv(f"file:/databricks/driver/{file}.csv", f"dbfs:/tmp/{file}.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC Due to the table only having chat messages, we can disable column level statistics for faster queries and streaming jobs.
# MAGIC Note: These settings should only be used when tuning specific performance of a table and not generally used.

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE Gaming.toxicity_chat SET TBLPROPERTIES
# MAGIC (
# MAGIC  'delta.checkpoint.writeStatsAsStruct' = 'false',
# MAGIC  'delta.checkpoint.writeStatsAsJson' = 'false'
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC Under the Data tab, A Gaming database should show and you should see 8 tables.
# MAGIC 
# MAGIC <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/delta-lake-tables.png"; width="25%" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Exploring the data
# MAGIC 
# MAGIC Toxicity tables Relationship Diagram
# MAGIC 
# MAGIC <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/toxicity-erd.png"; width="40%" />

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.1: Region group count of players & messages

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT region,
# MAGIC   count(distinct account_id) `# of players`,
# MAGIC   count(key) `# of messages`
# MAGIC FROM Gaming.Toxicity_chat
# MAGIC JOIN Gaming.Toxicity_players
# MAGIC ON Toxicity_chat.match_id = Toxicity_players.match_id
# MAGIC JOIN Gaming.Toxicity_match
# MAGIC ON Toxicity_match.match_id = Toxicity_players.match_id
# MAGIC JOIN Gaming.Toxicity_cluster_regions
# MAGIC ON Toxicity_match.cluster = Toxicity_cluster_regions.cluster
# MAGIC GROUP BY region
# MAGIC ORDER BY count(account_id) desc, count(account_id) desc

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.2: Number of messages sent per account

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT account_id,
# MAGIC   count(key) `# of messages` FROM Gaming.Toxicity_chat
# MAGIC JOIN Gaming.Toxicity_players
# MAGIC ON Toxicity_chat.match_id = Toxicity_players.match_id
# MAGIC AND Toxicity_chat.slot = Toxicity_players.player_slot
# MAGIC GROUP BY account_id
# MAGIC ORDER BY count(key) desc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * Build embedding and classification pipelines.

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2021]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Spark-nlp|Apache-2.0 License| https://nlp.johnsnowlabs.com/license.html | https://www.johnsnowlabs.com/
# MAGIC |Kaggle|Apache-2.0 License |https://github.com/Kaggle/kaggle-api/blob/master/LICENSE|https://github.com/Kaggle/kaggle-api|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|