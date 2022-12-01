# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# DBTITLE 1,How to implement and run this notebook
#When possible, always use one of the SHARED-XXX cluster available
#Tips: the %run cell defined above init a database with your name, and Python path variable is available, use it to store intermediate data or your checkpoints
#You need to run the setup cell to have these variable defined:
print(f"path={path}")
#Just save create the database to the current database, it's been initiliazed locally to your user to avoid conflict
print("your current database has been initialized to:")
print(sql("SELECT current_database() AS db").collect()[0]['db'])

# COMMAND ----------

# DBTITLE 1,Our Medallion Architecture Applied to Your Gas Turbine Use Case
# MAGIC %md-sandbox
# MAGIC - We'll be ingesting Turbine IOT sensor data (vibrations, speed etc) from our Power Plant (this notebook).
# MAGIC - With this data, we can build a ML model to predict when a turbine is defective and send a maintenance team to avoid outage
# MAGIC - We'll be able to build a Dashboard for the Business team to track our Power Plant efficiency and potential technical issues, with a estimation of the impact/gain to fix the equiment (in $)!

# COMMAND ----------

display_slide('137acXM8aj9dJUZbLpH3ZIorKHzBx3B0wqPRm-qdQ-yU', 38)  #hide this code

# COMMAND ----------

# MAGIC %md 
# MAGIC TODO: what's the data flow ? What will you do in more detail here for the Data Engineering part?
# MAGIC Tips: check this example: https://docs.google.com/presentation/d/137acXM8aj9dJUZbLpH3ZIorKHzBx3B0wqPRm-qdQ-yU/edit#slide=id.gf3af1439a3_0_54

# COMMAND ----------

display_slide('137acXM8aj9dJUZbLpH3ZIorKHzBx3B0wqPRm-qdQ-yU', 39)  #hide this code

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Our raw data is made available as files in a bucket mounted under /mnt/field-demos/manufacturing/iot_turbine/incoming-data-json
# MAGIC 
# MAGIC TODO: Use %fs to visualize the incoming data under /mnt/field-demos/manufacturing/iot_turbine/incoming-data-json

# COMMAND ----------

display(dbutils.fs.ls("/mnt/field-demos/manufacturing/iot_turbine"))

# COMMAND ----------

display(dbutils.fs.ls("/mnt/field-demos/manufacturing/iot_turbine/prediction"))

# COMMAND ----------

# DBTITLE 1,Take a look at the files on the Mount Point
display(dbutils.fs.ls("/mnt/field-demos/manufacturing/iot_turbine/incoming-data-json"))

# COMMAND ----------

# MAGIC %sql
# MAGIC create table turbine_power_generation as 
# MAGIC SELECT * FROM delta.`/mnt/field-demos/manufacturing/iot_turbine/power/`

# COMMAND ----------

# DBTITLE 1,Take a peek at the data and the schema
# MAGIC %sql
# MAGIC --TODO : Select and display the entire incoming json data using a simple SQL:
# MAGIC SELECT * FROM json.`/mnt/field-demos/manufacturing/iot_turbine/incoming-data-json`
# MAGIC 
# MAGIC -- What you have here is a list of IOT sensor metrics (AN1,AN2...) that you get from your turbine (vibration, speed...).
# MAGIC -- We'll then ingest these metrics and use them to detect when a turbine isn't healthy, so that we can prevent outage. 
# MAGIC 
# MAGIC -- TODO: take some time to understand the data, and keep it super simple (you can choose what AN1/2/3 represent for megacorp).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Bronze layer: ingesting incremental data as a stream

# COMMAND ----------

# DBTITLE 1,Stream landing files from cloud storage
# Set up the stream to begin reading incoming files from the mount point.

#TODO: ingest data using cloudfile
#Take some time to review the documentation: https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html#quickstart

#Incoming data is available under /mnt/field-demos/manufacturing/iot_turbine/incoming-data-json
#Goal: understand autoloader value and functionality (schema evolution, inference)
#What your customer challenges could be with schema evolution, schema inference, incremental mode having lot of small files? 
#How do you fix that ?

#load the stream with pyspark readStream and autoloader
#bronzeDF = spark.readStream .....
#Tips: use .option("cloudFiles.maxFilesPerTrigger", 1) to consume 1 file at a time and simulate a stream during the demo
#Tips: use your local path to save the scheam: .option("cloudFiles.schemaLocation", path+"/schema_bronze")    

#TODO: write the output as "turbine_bronze" delta table, with a trigger of 10 seconds or a trigger Once
# Write Stream as Delta Table
#bronzeDF.writeStream ...
#Tips: use your local path to save the checkpoint: .option("checkpointLocation", path+"/bronze_checkpoint")    
#Tips: if you're looking for your table in the data explorer, they're under the database having your name (see init cell)

# COMMAND ----------

# DBTITLE 1,Stream the JSON files into a Bronze Delta layer
(spark.readStream
      .format("cloudFiles")
      .option("cloudFiles.format", 'json')
      .option("cloudFiles.schemaLocation", path+"/schema_bronze")
      .load('/mnt/field-demos/manufacturing/iot_turbine/incoming-data-json')
      .writeStream
      .option("checkpointLocation", path+"/bronze_checkpoint")
      .table("turbine_bronze")
)

# COMMAND ----------

# DBTITLE 1,Take a peek at the freshly-created Bronze table
# MAGIC %sql
# MAGIC SELECT * from flightschool_akash_jaiswal.turbine_bronze

# COMMAND ----------

# DBTITLE 1,Set Table Properties to optimize data compaction and layout
# MAGIC %sql
# MAGIC -- TODO: which table property should you define to solve small files issue?
# MAGIC -- What's the typical challenge running streaming operation? And the value for your customer.
# MAGIC 
# MAGIC ALTER TABLE flightschool_akash_jaiswal.turbine_bronze
# MAGIC SET TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true,
# MAGIC                    delta.autoOptimize.autoCompact = true)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Silver layer: cleanup data and remove unecessary column

# COMMAND ----------

# MAGIC %md
# MAGIC #TODO: cleanup the silver table
# MAGIC Our bronze silver should have TORQUE with mostly NULL value and the _rescued column should be empty.#drop the TORQUE column, filter on _rescued to select only the rows without json error from the autoloader, filter on ID not null as you'll need it for your join later
# MAGIC 
# MAGIC from pyspark.sql.functions import col
# MAGIC 
# MAGIC silverDF = spark.readStream.table('turbine_bronze') ....
# MAGIC #TODO: cleanup the data, make sure all IDs are not null and _rescued is null (if it's not null it means we couldn't parse the json with the infered schema).
# MAGIC 
# MAGIC silverDF.writeStream ...
# MAGIC #TODO: write it back to your "turbine_silver" table

# COMMAND ----------

# DBTITLE 1,Bronze to Silver Transformation
from pyspark.sql.functions import col, isnan
import pyspark.sql.functions as F

silverDF = spark.readStream
    .table('turbine_bronze')\
    .where(F.col("ID").isNotNull() & ~F.col("_rescued_data").isNotNull())\
    .drop(F.col("TORQUE"))\
    .writeStream \
    .queryName("Bronze_to_Silver") \
    .format("delta")\
    .outputMode("append")\
    .option("checkpointLocation", path+"/silver_checkpoint")\
    .trigger(processingTime='10 seconds')\
    .table("turbine_silver")

# COMMAND ----------

# DBTITLE 1,Take a peek at the Silver table
# MAGIC %sql
# MAGIC SELECT * from flightschool_akash_jaiswal.turbine_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Gold layer: join information on Turbine status to add a label to our dataset

# COMMAND ----------

#TODO: the turbine status is available under /mnt/field-demos/manufacturing/iot_turbine/status as parquet file
# Use dbutils.fs to display the folder content

# COMMAND ----------

# DBTITLE 1,Take a look at the existing Parquet 'status' table
display(dbutils.fs.ls("/mnt/field-demos/manufacturing/iot_turbine/status"))

# COMMAND ----------

# DBTITLE 1,Display the parquet content using standard spark read
display(spark.read.format('parquet').load('dbfs:/mnt/field-demos/manufacturing/iot_turbine/status/'))

# COMMAND ----------

# MAGIC %sql
# MAGIC --TODO: save the status data as our turbine_status table
# MAGIC --Use databricks COPY INTO COMMAND https://docs.databricks.com/spark/latest/spark-sql/language-manual/delta-copy-into.html
# MAGIC --Tips: as of DBR 10.3, schema inference isn't available with cloudfile reading parquet. If you chose to use autoloader instead of the COPY INTO command to load the status, you'll have to specify the schema. 
# MAGIC COPY INTO turbine_status FROM ...

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table flightschool_akash_jaiswal.turbine_status

# COMMAND ----------

spark.sql("SET spark.databricks.delta.schema.autoMerge.enabled=True") 


# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS flightschool_akash_jaiswal.turbine_status;
# MAGIC 
# MAGIC COPY INTO flightschool_akash_jaiswal.turbine_status 
# MAGIC FROM "dbfs:/mnt/field-demos/manufacturing/iot_turbine/status"
# MAGIC FILEFORMAT = parquet

# COMMAND ----------

# DBTITLE 1,Join data with turbine status (Damaged or Healthy)
turbine_stream = spark.readStream.table('turbine_silver')
turbine_status = spark.read.table("turbine_status")

#TODO: do a left join between turbine_stream and turbine_status on the 'id' key and save back the result as the "turbine_gold" table
turbine_stream \
  .join(turbine_status.hint("broadcast"), "id", "left")\
  .writeStream\
  .queryName("silver_to_gold")\
  .format("delta")\
  .outputMode("append")\
  .option("checkpointLocation", path+"/gold_checkpoint")\
  .trigger(processingTime="10 seconds")\
  .table('turbine_gold')

# COMMAND ----------

# MAGIC %sql
# MAGIC --Our turbine gold table should be up and running!
# MAGIC select TIMESTAMP, AN3, SPEED, status from flightschool_akash_jaiswal.turbine_gold;

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Run DELETE/UPDATE/MERGE with DELTA ! 
# MAGIC We just realized that something is wrong in the data before 2020-05-10 00:00:00! Let's DELETE all this data from our gold table as we don't want to have wrong value in our dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO: DELETE all data from before '2020-01-01' in turbine_gold;

# COMMAND ----------

# MAGIC %sql
# MAGIC select timestamp, date(timestamp) from flightschool_akash_jaiswal.turbine_gold limit 100

# COMMAND ----------

# MAGIC %sql
# MAGIC USE DATABASE flightschool_akash_jaiswal;

# COMMAND ----------

# DBTITLE 1,Delete all records before 2020
# MAGIC %sql
# MAGIC DELETE FROM flightschool_akash_jaiswal.turbine_gold WHERE date(timestamp) < '2020-01-01' ;

# COMMAND ----------

# DBTITLE 1,Delta allows you to get the table modification history
# MAGIC %sql
# MAGIC DESCRIBE HISTORY flightschool_akash_jaiswal.turbine_gold;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grant Access to Database
# MAGIC 
# MAGIC Our data is now available. We can easily grant READ access to the Data Science and Data Analyst team!
# MAGIC *Note: (If on a Table-ACLs enabled High-Concurrency Cluster or using UC (coming soon!))*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Note: this won't work with standard cluster. 
# MAGIC -- DO NOT try to make it work during the demo (you need UC)
# MAGIC -- Understand what's required as of now (which cluster type) and the implications
# MAGIC -- explore Databricks Unity Catalog initiative (go/uc) 
# MAGIC 
# MAGIC GRANT SELECT ON DATABASE turbine_demo TO `data.scientist@databricks.com`
# MAGIC GRANT SELECT ON DATABASE turbine_demo TO `data.analyst@databricks.com`

# COMMAND ----------

# MAGIC %md ### What's next
# MAGIC 
# MAGIC TODO: wrapup on the Lakehouse concept. What have we done here?
# MAGIC What's the next step?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Don't forget to Cancel all the streams once your demo is over

# COMMAND ----------

for s in spark.streams.active:
  s.stop()
