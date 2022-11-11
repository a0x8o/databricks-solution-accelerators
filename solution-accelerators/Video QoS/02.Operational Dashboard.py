# Databricks notebook source
# MAGIC %run ./setup_config

# COMMAND ----------

sql('USE {}'.format(gold_database))
display(sql('SHOW TABLES IN {}'.format(gold_database)))

# COMMAND ----------

# MAGIC %md #Video Streaming Analytics

# COMMAND ----------

# MAGIC %md ###Unique Viewers - Last 24 hours

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select hour, count(distinct(user_id)) as viewers 
# MAGIC from video_activity
# MAGIC group by hour
# MAGIC order by hour

# COMMAND ----------

# MAGIC %md ###Views by Origin

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select origin, count(*) as count
# MAGIC from video_activity
# MAGIC group by origin 

# COMMAND ----------

# MAGIC %md #Video stats

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from video_activity

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select hour, video_id, count(*) as views
# MAGIC from mediaqosdemo_silver.player_events --changed from video_activity
# MAGIC where type = 'play'
# MAGIC group by video_id, hour
# MAGIC order by hour

# COMMAND ----------

# MAGIC %md ### Unique Views per Video

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select hour, video_id, count(distinct(user_id)) as unique_views
# MAGIC from mediaqosdemo_silver.player_events --changed from user activity
# MAGIC where type = 'play' 
# MAGIC group by video_id, hour
# MAGIC order by hour

# COMMAND ----------

# MAGIC %md ###Percentage of paused movies

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select video_id, type, count(*) as count
# MAGIC from mediaqosdemo_silver.player_events ----changed from user activity
# MAGIC where type = 'stop' or type = 'pause'
# MAGIC group by video_id, type

# COMMAND ----------

# MAGIC %md #Video Streaming Quality

# COMMAND ----------

# MAGIC %md ###Average bitrate

# COMMAND ----------

# MAGIC %sql
# MAGIC --to add resolution details
# MAGIC select hour, video_id, avg(avg_bitrate) / 1024 as avergage_bitrate
# MAGIC from video_activity
# MAGIC where type = 'stream'
# MAGIC group by video_id, hour
# MAGIC order by hour

# COMMAND ----------

# MAGIC %sql
# MAGIC --to add resolution details
# MAGIC 
# MAGIC select video_id, cdn_tracking_id as error
# MAGIC from video_activity
# MAGIC where type = 'error'

# COMMAND ----------

# MAGIC %sql
# MAGIC select * 
# MAGIC from video_activity 
# MAGIC where type = 'buffer' and buffer_type = "screenfreezedbuffer"

# COMMAND ----------

# MAGIC %sql
# MAGIC select video_id, avg(time_millisecond)/1000, count(*) as avg_buffer_time
# MAGIC from video_activity
# MAGIC where type = 'buffer' and buffer_type = "screenfreezedbuffer"
# MAGIC group by video_id

# COMMAND ----------

# MAGIC %sql
# MAGIC select video_id, avg(time_millisecond)/1000 as avg_buffer_time
# MAGIC from video_activity
# MAGIC where type = 'buffer' 
# MAGIC group by video_id

# COMMAND ----------

# MAGIC %md #Quality of the service using CDN Logs + Player Logs

# COMMAND ----------

# MAGIC %md Type of browser

# COMMAND ----------

# MAGIC %sql
# MAGIC select browserfamily as browser, count(*) 
# MAGIC from video_activity
# MAGIC group by browserfamily

# COMMAND ----------

# MAGIC %md First frame ( loading time ) for each AWS Cloudfornt edge node used to cache the video

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select location, resulttype,  avg(time_millisecond)/1000 as first_frame_time 
# MAGIC from video_activity
# MAGIC where type = 'firstframe' and location <> 'LHR50-C1'
# MAGIC group by location, resulttype

# COMMAND ----------

# MAGIC %md Buffer time per AWS edge location

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select location, avg(time_millisecond) 
# MAGIC from video_activity
# MAGIC where type = 'buffer' and buffer_type = "screenfreezedbuffer"
# MAGIC group by location

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC with mapping as (select requestip, location as country
# MAGIC from andrei.user_mapping)
# MAGIC select origin, location ,video_id, avg(time_millisecond)/1000 as average_time, count(*) as count
# MAGIC from video_activity
# MAGIC where type = 'buffer' and buffer_type = "screenfreezedbuffer"
# MAGIC group by video_id, location, origin 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from video_activity;

# COMMAND ----------

# MAGIC %sql
# MAGIC select origin, location, avg(time_millisecond)/1000 as average_time, count(*) as count
# MAGIC from video_activity
# MAGIC where type = 'firstframe' and  location <> 'LHR50-C1'
# MAGIC group by location, origin 

# COMMAND ----------

# MAGIC %md Loading time per Origin / Edge Node

# COMMAND ----------

# MAGIC %sql 
# MAGIC select location,video_id, avg(avg_bitrate) / 1024 
# MAGIC from video_activity
# MAGIC where type = 'stream'
# MAGIC group by location, video_id

# COMMAND ----------

# MAGIC %scala
# MAGIC // return the runId so the job runner can call the export API
# MAGIC val runId = dbutils.notebook.getContext.currentRunId match {
# MAGIC   case Some(runId) => runId.id.toString
# MAGIC   case None => ""
# MAGIC }
# MAGIC 
# MAGIC dbutils.notebook.exit(runId)
