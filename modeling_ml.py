#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def main(spark, train_file, val_file, model_file):

    train_df = spark.read.parquet(train_file)
    #train_df = spark.read.parquet('hdfs:/user/xx852/cf_train_small.parquet')
    val_df = spark.read.parquet(val_file)
    #val_df = spark.read.parquet('hdfs:/user/xx852/cf_val_small.parquet')
    train_df = train_df.select('user_label', 'track_label', 'count')
    val_df = val_df.select('user_label', 'track_label', 'count')
    train_df.cache()
    val_df.cache()
    # ALS for implicit feedback
    als = ALS(maxIter = 5, regParam = 0.01, implicitPrefs = True,
          userCol = 'user_label', itemCol = 'track_label', ratingCol = 'count')

    als_model = als.fit(train_df)

    val_preds = als_model.transform(val_df)
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='count', predictionCol='prediction')
    rmse = evaluator.evaluate(val_preds)
    print('default rmse:', rmse)

    # hyperparameter tuning
    ranks = [10, 25, 50, 100]
    reg_params = [0.01, 0.05, 0.10, 0.50, 1.0]
    alphas = [0.25, 0.50, 0.75, 1.00]
    #alphas = [0.25, 0.50]
    #ranks = [10,25]
    best_rank = None
    best_reg_param = None
    best_alpha = None
    best_model = None
    best_rmse = 100000 # set an arbitrarily large number
    #
    for rank_i, alpha_i, reg_param_i in itertools.product(ranks, alphas, reg_params):
        als = ALS(maxIter = 5, regParam = reg_param_i, implicitPrefs = True,alpha = alpha_i, 
                  rank =rank_i,userCol = 'user_label', itemCol = 'track_label', ratingCol = 'count')

        model = als.fit(train_df)
        val_preds  = model.transform(val_df)
        rmse = evaluator.evaluate(val_preds)
        print('Current rank:', rank_i)
        print('Current alpha:', alpha_i)
        print('Current reg:', reg_param_i) 
        if rmse < best_rmse:
            best_rank = rank_i
            best_reg_param = reg_param_i
            best_alpha = alpha_i
            best_model = model
            best_rmse = rmse 
    #         best_reg_param = reg_param
    #         best_alpha = alpha
    #         best_model = model
    #         best_rmse = rmse
    #
    print('Best rank:', best_rank)
    print('Best regParam:', best_reg_param)
    print('Best alpha:', best_alpha)
    print('Best rmse:', best_rmse)
    #
    # # save the best model
    best_model.save(model_file)


if __name__ == '__main__':

    spark = SparkSession.builder.appName('modeling').getOrCreate()

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    model_file = sys.argv[3]

    main(spark, train_file, val_file, model_file)

