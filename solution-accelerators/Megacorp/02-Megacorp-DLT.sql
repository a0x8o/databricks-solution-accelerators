-- Databricks notebook source
-- DBTITLE 1,Let's Build a Streaming Pipeline to Ingest and Visualize Turbine Sensor Data
-- MAGIC %md
-- MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/iot-wind-turbine/resources/images/iot-turbine-flow-3.png" width="1200" />

-- COMMAND ----------

-- DBTITLE 1,Introducing Delta Live Tables
-- MAGIC %md
-- MAGIC <img src="https://raw.githubusercontent.com/vadim/dlt-demo-slides/44b95dad5f85d5bb1a6284ad8a4c7715e121fcc7/images/dlt-00.png" width="1200"/>

-- COMMAND ----------

-- DBTITLE 1,Ingest Raw Data with Autoloader
-- MAGIC %md-sandbox
-- MAGIC <img src="https://raw.githubusercontent.com/vadim/dlt-demo-slides/44b95dad5f85d5bb1a6284ad8a4c7715e121fcc7/images/dlt-02.png" width="1200"/>

-- COMMAND ----------

-- DBTITLE 1,Our perspective on streaming and 
-- MAGIC %md-sandbox
-- MAGIC ### Join table to create gold layer available for Dashboarding & ML
-- MAGIC 
-- MAGIC <img src="https://www.databricks.com/wp-content/uploads/2021/09/DLT_graphic_tiers.jpg" width="1200"/>

-- COMMAND ----------

-- DBTITLE 1,Let's ingest the "raw" bronze-level sensor data, infer the schema and allow for schema evolution
CREATE STREAMING LIVE TABLE turbine_bronze_dlt (
  CONSTRAINT correct_schema EXPECT (_rescued_data IS NULL)
)
COMMENT "raw user data coming from json files ingested in incremental with Auto Loader to support schema inference and evolution"
TBLPROPERTIES ("quality" = "bronze")
AS SELECT * FROM cloud_files("/mnt/field-demos/manufacturing/iot_turbine/incoming-data-json",
                             "json",
                             map("cloudFiles.inferColumnTypes", "true"));

-- COMMAND ----------

-- DBTITLE 1,Declarative, Simple, and Powerful
-- MAGIC %md
-- MAGIC * Use intent-driven declarative development to abstract away the "how" and define "what to solve"
-- MAGIC * Automatically generate lineage based on table dependecies across the data pipeline
-- MAGIC * Automatically check for errors, missing dependencies and syntax errors

-- COMMAND ----------

-- DBTITLE 1,We can set Expectations and Constraints on the Data Source
CREATE STREAMING LIVE TABLE turbine_silver_dlt
  (CONSTRAINT valid_id EXPECT (ID IS NOT NULL)
  )
   COMMENT "Cleaned and improved turbine telemetry."
   TBLPROPERTIES ("quality" = "silver")
   AS
  SELECT
    AN3,
    AN4,
    AN5,
    AN6,
    AN7,
    AN8,
    AN9,
    ID,
    SPEED,
    TIMESTAMP
  FROM
    (SELECT * FROM STREAM(live.turbine_bronze_dlt)
      WHERE _rescued_data IS NULL
    ) AS table_abc

-- COMMAND ----------

-- DBTITLE 1,We can work with streaming and static tables at the same time
CREATE LIVE TABLE turbine_status_dlt
  COMMENT "Ingest Gas turbine status information"
  TBLPROPERTIES ("quality" = "mapping")
  AS SELECT * FROM parquet.`/mnt/field-demos/manufacturing/iot_turbine/status`;

-- COMMAND ----------

-- DBTITLE 1,Join in the status data to create gold layer available for Dashboarding & ML
CREATE STREAMING LIVE TABLE turbine_gold_dlt
  COMMENT 'Join the streaming sensor data with the status table'
  TBLPROPERTIES ('quality' = 'gold')
  AS SELECT
    a.AN3,
    a.AN4,
    a.AN5,
    a.AN6,
    a.AN7,
    a.AN8,
    a.AN9,
    a.ID,
    a.SPEED,
    a.TIMESTAMP,
    b.status
  FROM STREAM(LIVE.turbine_silver_dlt) AS a
    LEFT JOIN LIVE.turbine_status_dlt AS b
    ON a.ID = b.ID

-- COMMAND ----------

-- DBTITLE 1,Example Data Quality dashboard
-- MAGIC %md
-- MAGIC # Checking your data quality metrics with Delta Live Tables
-- MAGIC Delta Live Tables tracks all your data quality metrics. You can leverage the expecations directly as SQL table with Databricks SQL to track your expectation metrics and send alerts as required. This let you build a dashboard such as this one!
-- MAGIC 
-- MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/retail-dlt-data-quality-dashboard.png">
-- MAGIC 
-- MAGIC <a href="https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/6f73dd1b-17b1-49d0-9a11-b3772a2c3357-dlt---retail-data-quality-stats?o=1444828305810485" target="_blank">Data Quality Dashboard</a>
