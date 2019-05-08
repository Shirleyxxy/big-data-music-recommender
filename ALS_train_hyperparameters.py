#!/usr/bin/env python
# -*- coding: utf-8 -*-

# module load python/gnu/3.6.5
# module load spark/2.4.0
# PYSPARK_PYTHON=$(which python) pyspark

# To run: 
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
    val_preds = als_model.transform(val_df)
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='count', predictionCol='prediction')
    rmse = evaluator.evaluate(val_preds)
    print('Baseline Root-mean-square error = ', rmse)
    # Baseline Root-mean-square error =  7.652006276502559
    
    
    # Generate top 10 movie recommendations for each user
    userRecs = als_model.recommendForAllUsers(10)
    
    # Hyper-parameters tunning 
    ranks = [10, 20, 40, 80]
    reg_params = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    alphas = [0.25, 0.50, 0.75, 1.00]
    best_rank = None
    best_reg_param = None
    best_alpha = None
    best_model = None
    best_rmse = 1e9
    
    for rank, alpha, reg_param in itertools.product(ranks, alphas, reg_params):
        als = ALS(maxIter = 5, regParam = reg_param, implicitPrefs = True, alpha = alpha, rank =rank, userCol = 'user_label', itemCol = 'track_label', ratingCol = 'count')

        model = als.fit(train_df)
        val_preds  = model.transform(val_df)
        rmse = evaluator.evaluate(val_preds)
        print('Current rank:', rank)
        print('Current alpha:', alpha)
        print('Current reg:', reg_param) 
        if rmse < best_rmse:
            best_rank = rank
            best_reg_param = reg_param
            best_alpha = alpha
            best_model = model
            best_rmse = rmse 

    print('Best rank:', best_rank)
    print('Best regParam:', best_reg_param)
    print('Best alpha:', best_alpha)
    print('Best rmse:', best_rmse)
    
    # save the best model
    best_model.save(model_file)

    
#     ####### Archived work using MLLiB ###########
    
#     # Try with sample of 10% of data first
#     df = df.sample(False, 0.1)
    
#     # A Ratings object is made up of (user, item, rating)
#     ratings = df.rdd.map(lambda x: Rating(int(x[4]), int(x[5]), float(x[1])))
    
#     #Setting up the parameters for ALS
#     rank = 5 # Latent Factors to be made
#     numIterations = 10 # Times to repeat process
#     #Create the model on the training data
#     model = ALS.trainImplicit(ratings, rank, numIterations)
    
#     model.save(spark.SparkContext, model_file)

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

