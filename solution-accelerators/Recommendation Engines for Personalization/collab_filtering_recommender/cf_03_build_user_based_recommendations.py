# Databricks notebook source
# MAGIC %md The purpose of this notebook is to build and evaluate user-based collaborative filtering recommendations.  This notebook is designed to run on a **Databricks 7.1+ cluster**.

# COMMAND ----------

# MAGIC %md # Introduction 
# MAGIC With a basis for performing user-matching in place, let's build a collaborative filter leveraging product-purchase similarities between users:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_userbasedcollab.gif" width="300">

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSH

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, max, collect_list, lit, expr, coalesce, pow, sum
from pyspark.sql.types import *

import pandas as pd
import numpy as np

import math
import shutil

# COMMAND ----------

# MAGIC %md # Step 1: Build a Recommendation
# MAGIC 
# MAGIC To build a recommendation, we'll need to first reconstruct the ratings vectors and LSH dataset we explored in the last notebook.  Let's do this now:

# COMMAND ----------

# DBTITLE 1,Assemble Ratings Vectors
# define and register UDF for vector construction
@udf(VectorUDT())
def to_vector(size, index_list, value_list):
    ind, val = zip(*sorted(zip(index_list, value_list)))
    return Vectors.sparse(size, ind, val)
    
_ = spark.udf.register('to_vector', to_vector)

# generate ratings vectors 
ratings_vectors = spark.sql('''
  SELECT 
      user_id,
      to_vector(size, index_list, value_list) as ratings,
      size,
      index_list,
      value_list
    FROM ( 
      SELECT
        user_id,
        (SELECT max(product_id) + 1 FROM instacart.products) as size,
        COLLECT_LIST(product_id) as index_list,
        COLLECT_LIST(normalized_purchases) as value_list
      FROM ( -- all users, ratings
        SELECT
          user_id,
          product_id,
          normalized_purchases
        FROM instacart.user_ratings
        WHERE split = 'calibration'
        )
      GROUP BY user_id
      )
    ''')

# COMMAND ----------

# MAGIC %md **Note** The *bucketLength* and *numHashTable* settings used here differ from those explored in the last notebook.  These were arrived at by examining performance and evaluation metrics results (as shown below).

# COMMAND ----------

# DBTITLE 1,Build LSH Dataset
bucket_length = 0.0025
lsh_tables = 5

lsh = BucketedRandomProjectionLSH(
  inputCol = 'ratings', 
  outputCol = 'hash', 
  numHashTables = lsh_tables, 
  bucketLength = bucket_length
  )

# fit the algorithm to the dataset
fitted_lsh = lsh.fit(ratings_vectors)

# assign LSH buckets to users
hashed_vectors = (
  fitted_lsh
    .transform(ratings_vectors)
    ).cache()

hashed_vectors.createOrReplaceTempView('hashed_vectors')

# COMMAND ----------

# MAGIC %md Let's now consider how we might build recommendations for a given customer.  Using values for one of our customers, *i.e.* user_id 148, we can assemble a vector of product preferences:

# COMMAND ----------

# DBTITLE 1,Retrieve Demonstration User Vector
user_148 = (
  hashed_vectors
    .filter('user_id=148')
    .select('user_id','ratings')
  )

user_148.collect()[0]['ratings']

# COMMAND ----------

# MAGIC %md For this customer, we need to identify similar customers from which we might derive recommendations.  One way to do this is to specify a target number of users that are most similar to a given user:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_nearestneighbors.gif" width="300">
# MAGIC 
# MAGIC This *nearest neighbors* approach has the benefit of returning a consistent number of users from which to build ratings but the similarity between those users and our given user can vary:
# MAGIC 
# MAGIC **NOTE** The user itself, *i.e.* user_id 148, is included in the result set.  If we wish to retrieve 10 *other* users, we would need to specify 11 nearest neighbors and exclude our user from the results afterwards.

# COMMAND ----------

# DBTITLE 1,Retrieve Top 10 Most Similar Users
number_of_customers = 10

# retrieve n nearest customers 
similar_k_users = (
  fitted_lsh.approxNearestNeighbors(
    hashed_vectors, 
    user_148.collect()[0]['ratings'], # must be a vector value (not a dataframe)
    number_of_customers, 
    distCol='distance'
    )
    .select('user_id', 'distance')
  )
  
display(similar_k_users)

# COMMAND ----------

# MAGIC %md Another way to tackle the problem of identifying similar users to define a maximum distance from a given user and select all users within that range:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_distance.gif" width="400">
# MAGIC 
# MAGIC With this approach we receive users with a more predictable degree of similarity but have to deal with a variable number of users (if any) being returned.  We also need to explore our dataset to understand which distance *thresholds* give us the desired number of users on a consistent basis:
# MAGIC 
# MAGIC **NOTE** The *approxSimilarityJoin()* method requires our target user to be submitted as a dataframe. The benefit of this is that we can supply more than one user in our target user dataset (not shown here but employed in later notebooks).

# COMMAND ----------

# DBTITLE 1,Retrieve Users in a Distance Range
max_distance_from_target = 1.3

# retreive all users within a distance range
similar_d_users = (
    fitted_lsh.approxSimilarityJoin(
      user_148,
      hashed_vectors,  
      threshold = max_distance_from_target, 
      distCol='distance'
      )
    .selectExpr('datasetA.user_id as user_a', 'datasetB.user_id as user_b', 'distance')
    .orderBy('distance', ascending=True)
    )
  
display(similar_d_users)

# COMMAND ----------

# MAGIC %md Regardless of how we select the *similar* users, it's important we understand how similarity is calculated.  In a vector space such as this, similarity is often based on the distance between two vectors. There are several ways to calculate distance but Euclidean distance is one of the more popular of these.
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_distance.png" width="400">
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_basicsimilarity.png" width="350">
# MAGIC 
# MAGIC The further the distance between two users, the more dissimilar they are.  In this regard, we can think of similarity as being the inverse of the distance itself.  But because distance can be zero, we often modify the relationship by adding 1 to distance in order to avoid division by zero errors.  
# MAGIC 
# MAGIC There are many other ways to calculate similarities but this seems to work well for our purposes:

# COMMAND ----------

# DBTITLE 1,Calculate Similarity for Similar Users
# calculate similarity score
similar_users = (
  similar_k_users
    .withColumn('similarity', lit(1) / (lit(1) + col('distance')))
  )

display(similar_users)

# COMMAND ----------

# MAGIC %md In an L2-normalized vector space, the minimum distance between two points is 0.0 while the maximum distance is equal to the square-root of two.  This translates into a maximum potential similarity score of 1.0 and a minimum similarity score of 0.414.  Applying a standard [min-max transformation](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization), we can convert our similiarity scores to fall within the range 1.0 (most similar) and 0.0 (least similar):
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_similarityminmax2.png" width="600">

# COMMAND ----------

# DBTITLE 1,Apply Min-Max Transformation to Similarity Score
# calculate lowest possible unscaled similarity score
min_score = 1 / (1 + math.sqrt(2))

# calculate similarity score
similar_users = (
  similar_users
    .withColumn(
       'similarity_rescaled', 
       (col('similarity') - lit(min_score)) / lit(1.0 - min_score)
       )
     )

# make available for SQL query
similar_users.createOrReplaceTempView('similar_users')

display(similar_users)

# COMMAND ----------

# MAGIC %md Rescaling our similarity scores makes it easier for us to judge the degree of similarity between users. To calculate a recommendation, the similarity score will be used as a weight in a weighted average of product preferences/ratings.  There's no reason we can't make further adjustments to our similarity calculations, such as squaring the rescaled similarity scores so that weight increases exponentially as similarity approaches 1.0 Such adjustments allow us to adjust the degree of influence that similarity has on our recommendations.
# MAGIC 
# MAGIC In calculating recommendation scores, it's important to recognize that the lack of a rating/preference for a product by a given user may be used to imply a rating/preference of 0.0.  Depending on our objectives, we might apply such logic or choose to skip it (which would have the effect of broadening the range of products recommended to users).  Here, we can see which products are being brought into our recommendations (by which similar user) with which implied ratings and similarity scores:

# COMMAND ----------

# DBTITLE 1,Product Ratings from Similar Users
similar_ratings = spark.sql('''
      SELECT
        m.user_id,
        m.product_id,
        COALESCE(n.normalized_purchases, 0.0) as normalized_purchases,
        m.similarity_rescaled
      FROM ( -- get complete list of products across similar users
        SELECT
          x.user_id,
          y.product_id,
          x.similarity_rescaled
        FROM (
          SELECT user_id, similarity_rescaled
          FROM similar_users
          ) x
        CROSS JOIN instacart.products y
        ) m
      LEFT OUTER JOIN ( -- retrieve ratings actually provided by similar users
        SELECT x.user_id, x.product_id, x.normalized_purchases 
        FROM instacart.user_ratings x 
        LEFT SEMI JOIN similar_users y 
          ON x.user_id=y.user_id 
        WHERE x.split = 'calibration'
          ) n
        ON m.user_id=n.user_id AND m.product_id=n.product_id
      ''')

display(similar_ratings)

# COMMAND ----------

# MAGIC %md Using these ratings and our similarity scores, we can now calculate per-product weighted averages, averaging the similarity weighted ratings from our similar users:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_ratingcalc.png" width="600">

# COMMAND ----------

# DBTITLE 1,Calculate Recommendation Scores
product_ratings = ( 
   similar_ratings
    .groupBy('product_id')
      .agg( 
        sum(col('normalized_purchases') * col('similarity_rescaled')).alias('weighted_rating'),
        sum('similarity_rescaled').alias('total_weight')
        )
    .withColumn('recommendation_score', col('weighted_rating')/col('total_weight'))
    #.select('product_id', 'recommendation_score')
    .orderBy('recommendation_score', ascending=False)
  )

display(product_ratings)

# COMMAND ----------

# MAGIC %md To see the influence of similar users on product recommendations, let's retrieve ratings for our target user, *user_id 148*.  Comparing the implied preferences of this user with the recommendation scores shows the influence of similar users on the recommendations:

# COMMAND ----------

# DBTITLE 1,Compare Recommendation Scores to User-Implied Scores
# retreive actual ratings from this user
user_product_ratings = (
  spark
    .table('instacart.user_ratings')
    .filter("user_id = 148 and split = 'calibration'")
  )

# combine with recommender ratings
product_ratings_for_user = (
    product_ratings
      .join( user_product_ratings, on='product_id', how='outer')
      .selectExpr('product_id', 'COALESCE(normalized_purchases, 0.0) as user_score', 'recommendation_score')
      .orderBy('recommendation_score', ascending=False)
  )

display(product_ratings_for_user)

# COMMAND ----------

# MAGIC %md In comparing the user's actual scores to the recommendation scores, it's interesting to see how some recommendations align with the user's scores, how others are pushed higher or lower based on the input of similar users, and how new products are introduced through recommendations (as indicated by a user score of 0.0 and a recommendation score of greater than 0.0). There were numerous little choices that went into deciding which users contributed to these recommendations and how similarities were applied.  And there are still more options such as assigning scores to products purchased by the target user as-is with no consideration of similar users' ratings of those products or the simple exclusion of previously purchased products in order to recommend new only items.  The key consideration in navigating these choices is the business outcome we wish to drive through the recommender.

# COMMAND ----------

# MAGIC %md # Step 2: Evaluate Recommendations
# MAGIC 
# MAGIC At this point, we've got the mechanics for calculating user-based collaborative filter recommendations addressed, but are the recommendations we're producing any good? It is a really important point to consider when working with recommenders, **what does *good* mean?**
# MAGIC 
# MAGIC [Gunawardana and Shani](https://www.jmlr.org/papers/volume10/gunawardana09a/gunawardana09a.pdf) provide an excellent examination of this question. They argue that recommenders exist to address specific goals and should be assessed in terms of their ability to meet that goal.  So, **what's the *goal* of our recommender?**
# MAGIC 
# MAGIC In a grocery scenario such as the one captured by our dataset, our goal is most likely to present users with choices for grocery items they may be receptive to buying in order to:
# MAGIC 
# MAGIC 1. Enable a more efficient shopping experience,
# MAGIC 2. Encourage the customer to buy new products, *i.e.* products they have not purchased from us in the past,
# MAGIC 3. Encourage the customer to buy old products, *i.e.* products they have purchased from us in the past, but which they had not originally intended to purchase when coming to the application, site or store
# MAGIC 
# MAGIC The influence of our recommender on each of these outcomes is easily measured in an experimental setup where some users are exposed to the recommendations while some are not and we compare the metrics aligned with the goal.  For example, if our goal is to enable a faster shopper experience, do customers who are exposed to our recommender complete their journey from entrance to checkout faster than those who are not?  If our goal is to encourage customers to buy new products, do customers exposed to our recommender buy recommended products not found in their previous buying history at higher rates than those not exposed?  If our goal is to encourage buying previously purchased products that may have not provided the impetus for their initial engagement, do customers exposed to our recommender having larger overall basket sizes (limited to previously bought products) than those not exposed?
# MAGIC 
# MAGIC While conceptually straightforward to measure, the implementation of such an (online) experiment requires careful planning and implementation and carries with it an opportunity cost not to mention the risk of delivering a bad customer experience which negatively impacts the customer's perception of our brand. For these reasons, we often substitute our real world goals for the recommender with a proxy goal which can be evaluated in an offline manner before attempting an experimental rollout.
# MAGIC 
# MAGIC In defining a proxy goal for offline evaluation, it maybe helpful to carefully reflect on exactly what our recommender scores represent.  In the scenario presented in these workbooks, our scores are weighted averages of an implied rating derived from repeat purchases.  [Hu, Koren & Volinsky](http://yifanhu.net/PUB/cf.pdf) provide a really nice exploration of implicit ratings derived from consumption, and while their work is focused on media, we can piggy-back on their idea of scores as ranking mechanisms and evaluate our recommender in terms of the average position within the rankings that users respond to.  This metric provides a nice offline check that our recommenders align with what customers eventually buy.
# MAGIC 
# MAGIC But before we do that, we need to generate a set of recommendations with which we can perform our evaluation.  Because of the computational intensity of calculating recommendations for all users in our dataset, we'll limit our evaluation to a random sampling of customers:
# MAGIC 
# MAGIC **NOTE** Even with the limiting of our evaluation to a small percentage of customers, the following steps are computationally intensive. We've adjusted the shuffle partition count for the size of the cluster with which we are working.  See [this document](https://docs.microsoft.com/en-us/azure/architecture/databricks-monitoring/performance-troubleshooting#common-performance-bottlenecks) for more info on this and other performance tuning mechanisms.

# COMMAND ----------

# DBTITLE 1,Alter Shuffle Partition Count
max_partition_count = sc.defaultParallelism * 100
spark.conf.set('spark.sql.shuffle.partitions', max_partition_count) 

# COMMAND ----------

# DBTITLE 1,Get Similar Users for a Random Sample of Users
# ratio of customers to sample
sample_fraction = 0.10

# calculate max possible distance between users
max_distance = math.sqrt(2)

# calculate min possible similarity (unscaled)
min_score = 1 / (1 + math.sqrt(2))

# remove any old comparisons that might exist
shutil.rmtree('/dbfs/mnt/instacart/gold/similarity_results', ignore_errors=True)

# perform similarity join for sample of users
sample_comparisons = (
  fitted_lsh.approxSimilarityJoin(
    hashed_vectors.sample(withReplacement=False, fraction=sample_fraction), # use a random sample for our target users
    hashed_vectors,
    threshold = max_distance,
    distCol = 'distance'
    )
    .withColumn('similarity', lit(1)/(lit(1)+col('distance')))
    .withColumn('similarity_rescaled', (col('similarity') - lit(min_score)) / lit(1.0 - min_score))
    .selectExpr(
      'datasetA.user_id as user_a',
      'datasetB.user_id as user_b',
      'similarity_rescaled as similarity'
      )
  )

# write output for reuse
(
  sample_comparisons
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/gold/similarity_results')
  )

display(
  spark.table( 'DELTA.`/mnt/instacart/gold/similarity_results`' )
  )

# COMMAND ----------

# MAGIC %md In performing the similarity join, we set the threshold to the maximum possible distance between users.  In effect, we are grabbing all similar users per LSH, allowing us to filter to a fixed number of similar users in the steps below.  In this way, we are emulating what is performed with the *approxNearestNeighbors()* method call:

# COMMAND ----------

# DBTITLE 1,Get k Similar Users
number_of_customers = 10

# get k number of similar users for each sample user
similar_users =  (
    spark.sql('''
      SELECT 
        user_a, 
        user_b, 
        similarity
      FROM (
        SELECT
          user_a,
          user_b,
          similarity,
          ROW_NUMBER() OVER (PARTITION BY user_a ORDER BY similarity DESC) as seq
        FROM DELTA.`/mnt/instacart/gold/similarity_results`
        )
      WHERE seq <= {0}
      '''.format(number_of_customers)
      )
    )

similar_users = similar_users.createOrReplaceTempView('similar_users')
display(similar_users)

# COMMAND ----------

# MAGIC %md Using our dataset of similar users, we can now assemble ratings and recommendations in a manner similar to how we did in the prior step:

# COMMAND ----------

# DBTITLE 1,Retrieve Per-User Product Ratings
similar_ratings = spark.sql('''
    SELECT
      m.user_a,
      m.user_b,
      m.product_id,
      COALESCE(n.normalized_purchases, 0.0) as normalized_purchases,
      m.similarity
    FROM (
      SELECT
        x.user_a,
        x.user_b,
        y.product_id,
        x.similarity
      FROM similar_users x
      CROSS JOIN instacart.products y
      ) m
    LEFT OUTER JOIN ( -- retrieve ratings actually provided by similar users
      SELECT 
        user_id as user_b, 
        product_id, 
        normalized_purchases 
      FROM instacart.user_ratings
      WHERE split = 'calibration'
        ) n
      ON m.user_b=n.user_b AND m.product_id=n.product_id
      ''')

display(similar_ratings)

# COMMAND ----------

# DBTITLE 1,Generate Per-User Recommendations
product_ratings = ( 
   similar_ratings
    .groupBy('user_a','product_id')
      .agg( 
        sum(col('normalized_purchases') * col('similarity')).alias('weighted_rating'),
        sum('similarity').alias('total_weight')
        )
    .withColumn('recommendation_score', col('weighted_rating')/col('total_weight'))
    .select('user_a', 'product_id', 'recommendation_score')
    .orderBy(['user_a','recommendation_score'], ascending=[True,False])
  )

product_ratings.createOrReplaceTempView('product_ratings')

display(
  product_ratings
  )  

# COMMAND ----------

# MAGIC %md In keeping with Hu *et al.*, we can now convert our recommendations to percentile rankings (*rank_ui*) with 0.0% representing the top most preferred recommendation for each customer.  Here is the unit of code that calculates the percentile rankings, presented separately for review:

# COMMAND ----------

# DBTITLE 1,Convert Recommendations to Percent Ranks
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   user_a as user_id,
# MAGIC   product_id,
# MAGIC   recommendation_score,
# MAGIC   PERCENT_RANK() OVER (PARTITION BY user_a ORDER BY recommendation_score DESC) as rank_ui
# MAGIC FROM product_ratings
# MAGIC ORDER BY user_a, recommendation_score DESC

# COMMAND ----------

# MAGIC %md But why use a percent rank?  First, converting our recommendation score to a rank allows us to sort our recommendations from highest to lowest without regard for the raw scores.  If are top-most recommended product has a weak recommendation score because there simply aren't many other customers like our given customer, that product is still the top-most recommended product.  Second, by converting the ranks to a **percent rank**, we put the ranks on a standard scale from 0.0 (most recommended) to 1.0 (least recommended).  This allows us to calculate an overall evaluation metric which expresses where along the standard scale our customers are landing with regard to our recommendations:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_rankingseval.gif" width="400">
# MAGIC 
# MAGIC Along this scale, we'd expect that if a customer is purchasing items with an average percent rank of around 50% (0.5), then our recommendations are really no better than random suggestions.  If we average greater than 50%, we're making recommendations in a direction that's opposed to where our customer is actually going.  But if the average percent rank is less than 50%, we're recommending products to our customer in a manner that aligns with their recognized preferences with that alignment continuing to improve as we push the average closer and closer to 0.0.
# MAGIC 
# MAGIC And here is how our recommendations, built using information from our calibration period align with actual purchases in our evaluation period:

# COMMAND ----------

# DBTITLE 1,Evaluate User-Based Recommendations
eval_set = (
  spark
    .sql('''
    SELECT 
      x.user_id,
      x.product_id,
      x.r_t_ui,
      y.rank_ui
    FROM (
      SELECT
        user_id,
        product_id,
        normalized_purchases as r_t_ui
      FROM instacart.user_ratings 
      WHERE split = 'evaluation' -- the test period
        ) x
    INNER JOIN (
      SELECT
        user_a as user_id,
        product_id,
        PERCENT_RANK() OVER (PARTITION BY user_a ORDER BY recommendation_score DESC) as rank_ui
      FROM product_ratings
      ) y
      ON x.user_id=y.user_id AND x.product_id=y.product_id
      ''').cache()
  )

display(
  eval_set
    .withColumn('weighted_r', col('r_t_ui') * col('rank_ui') )
    .groupBy()
      .agg(
        sum('weighted_r').alias('numerator'),
        sum('r_t_ui').alias('denominator')
        )
    .withColumn('mean_percent_rank', col('numerator')/col('denominator'))
    .select('mean_percent_rank')
  )

# COMMAND ----------

# MAGIC %md The lower the mean ranking the more customers are purchasing products from higher in our recommended product list.  Another way to think about this metric is that the lower the score, the lower the number of *junk* recommendations the user has to wade through to get to what they want. The value returned here would indicate that our recommendations align very well with those purchases, and there's likely a good reason for this.
# MAGIC 
# MAGIC When we constructed our ratings we decided to keep our users implied ratings in the mix.  This means that things that users have expressed preferences for in the past will sit higher in our recommendations than other products.  In a grocery shopping scenario, where customers very typically establish a pattern of repeat product purchases, this works well, especially if our goal is to help them get what they want and complete the transaction. 
# MAGIC 
# MAGIC But is this always the right way to approach recommendations, even in a grocery scenario?  Consider product categories such as craft beer, where novelty and surprise are highly regarded. Simply recommending what's been purchased in the past may give your customers the sense that your offering are less diverse and appealing than they might otherwise be.  We need to carefully consider our goals and the proper way to evaluate them, no matter how *good* our metrics may appear to be. 
# MAGIC 
# MAGIC Before moving on, we wanted to take a moment to compare our recommendations to a popular strategy of simply recommending the most popular products to customers.  Here, we calculate the mean percent rank metric for these products:

# COMMAND ----------

# DBTITLE 1,Rank Popular Product Recommendations
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   product_id,
# MAGIC   PERCENT_RANK() OVER (ORDER BY normalized_purchases DESC) as rank_ui
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.product_id,
# MAGIC     COALESCE(y.normalized_purchases,0.0) as normalized_purchases
# MAGIC   FROM (SELECT product_id FROM instacart.products) x
# MAGIC   LEFT OUTER JOIN instacart.naive_ratings y
# MAGIC     ON x.product_id=y.product_id
# MAGIC   WHERE split = 'calibration'
# MAGIC   )

# COMMAND ----------

# MAGIC %md And using those percent ranks, we can calculate our evaluation metric against the evaluation set:

# COMMAND ----------

# DBTITLE 1,Evaluate Popular Product Recommendations
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   SUM(r_t_ui * rank_ui) / SUM(rank_ui) as mean_percent_rank
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     x.user_id,
# MAGIC     x.product_id,
# MAGIC     x.r_t_ui,
# MAGIC     y.rank_ui
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       p.user_id,
# MAGIC       p.product_id,
# MAGIC       p.normalized_purchases as r_t_ui
# MAGIC     FROM instacart.user_ratings p
# MAGIC     INNER JOIN (SELECT DISTINCT user_a as user_id FROM similar_users) q
# MAGIC       ON p.user_id=q.user_id
# MAGIC     WHERE p.split = 'evaluation' -- the test period
# MAGIC       ) x
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       product_id,
# MAGIC       PERCENT_RANK() OVER (ORDER BY normalized_purchases DESC) as rank_ui
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         x.product_id,
# MAGIC         COALESCE(y.normalized_purchases,0.0) as normalized_purchases
# MAGIC       FROM (SELECT product_id FROM instacart.products) x
# MAGIC       LEFT OUTER JOIN instacart.naive_ratings y
# MAGIC         ON x.product_id=y.product_id
# MAGIC       WHERE split = 'calibration'
# MAGIC       )
# MAGIC     ) y
# MAGIC     ON x.product_id=y.product_id
# MAGIC     )

# COMMAND ----------

# MAGIC %md The results of recommending the most popular products are really interesting.  The 30% mean percent rank tells us that we're doing better than making random suggestions but we're not doing as well as when we consider users and their most similar peers.  The likely reason based on what we saw earlier with our data is that users tend to buy a small subset of the products we offer and they tend to lock into patterns of buying those same products over and over.  If those products don't significantly overlap between users (which you'd expect in a grocery scenario or why else would grocery stores stock the diversity of products they do), you'd expect very few *typical* shoppers and instead want to recognize the individual preferences users express through their actual buying patterns.

# COMMAND ----------

# DBTITLE 1,Remove Cached Objects
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()
