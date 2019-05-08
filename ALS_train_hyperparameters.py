#!/usr/bin/env python
# -*- coding: utf-8 -*-

# module load python/gnu/3.6.5
# module load spark/2.4.0
# PYSPARK_PYTHON=$(which python) pyspark

# To run: 
#full train: spark-submit ALS_train.py cf_train_transformed_v8.parquet cf_val_transformed_v1.parquet hdfs:/user/nhl256/als_train.model

# spark-submit ALS_train.py hdfs:/user/nhl256/cf_val_transformed.parquet hdfs:/user/nhl256/cf_test_transformed.parquet hdfs:/user/nhl256/train_als_val_small.model


import sys
import itertools
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
# from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


def main(spark, train_file, val_file, model_file):
    
    train = spark.read.parquet(train_file)
    val = spark.read.parquet(val_file)
    
    # Train and evaluate a model 
    
    train_df = train.select('user_label', 'track_label', 'count')
    val_df = val.select('user_label', 'track_label', 'count')
    
    als = ALS(maxIter = 5, regParam = 0.01, implicitPrefs = True, userCol = 'user_label', itemCol = 'track_label', ratingCol = 'count')
    
    als_model = als.fit(train_df)
    
    
    # Generate top 10 song recommendations for each user 
    predictions = als_model.recommendForAllUsers(10)
    prediction_df = predictions.rdd.map(lambda r: (r.user_label, [i[0] for i in r.recommendations])).toDF()
    prediction_df = prediction_df.selectExpr("_1 as user_label", "_2 as recommendations")
    
    # Get the songs that each user listen to 
    val_grouped = test_df.groupBy('user_label').agg(F.collect_list(F.col('track_label')).alias('track_label'))
    
    # Join table
    val_pred = val_grouped.join(prediction_df, "user_label", "inner")
    
    # Instantiate regression metrics to compare predicted and actual ratings
    rdd = val_pred.select('recommendations', 'track_label').rdd
    ranking_metrics = RankingMetrics(rdd)

    # MAP
    print("MAP = %s" % ranking_metrics.meanAveragePrecision)
    
    

    
    ################## Hyper-parameters tunning ##################
    ranks = [10, 20]
    reg_params = [0.001, 0.005]
    alphas = [0.25]
    best_rank = None
    best_reg_param = None
    best_alpha = None
    best_model = None
    best_map = 1e9
    
    for rank, alpha, reg_param in itertools.product(ranks, alphas, reg_params):
        als = ALS(maxIter = 5, regParam = reg_param, implicitPrefs = True, alpha = alpha, rank =rank, userCol = 'user_label', itemCol = 'track_label', ratingCol = 'count')

        model = als.fit(train_df)
        # Look into top 100 songs recommended
        predictions = model.recommendForAllUsers(100)
        prediction_df = predictions.rdd.map(lambda r: (r.user_label, [i[0] for i in r.recommendations])).toDF()
        prediction_df = prediction_df.selectExpr("_1 as user_label", "_2 as recommendations")
        
        
        # Join table
        true_pred = val_grouped.join(prediction_df, "user_label", "inner")
    
        # Instantiate regression metrics to compare predicted and actual ratings
        rdd = val_pred.select('recommendations', 'track_label').rdd
        ranking_metrics = RankingMetrics(rdd)

        # MAP
        map = ranking_metrics.meanAveragePrecision
    
        print('Current rank:', rank)
        print('Current alpha:', alpha)
        print('Current reg:', reg_param) 
        if map < best_map:
            best_rank = rank
            best_reg_param = reg_param
            best_alpha = alpha
            best_model = model
            best_map = map 

    print('Best rank:', best_rank)
    print('Best regParam:', best_reg_param)
    print('Best alpha:', best_alpha)
    print('Best map:', best_map)
    
    # save the best model
    best_model.save(model_file)

    

if __name__ == '__main__':
    
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS_train').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    val_file = sys.argv[2]

    # And the location to store the trained model
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, val_file, model_file)

