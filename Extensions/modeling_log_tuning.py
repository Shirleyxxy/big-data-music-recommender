#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F


def main(spark, train_file, val_file, model_file):

    train_df = spark.read.parquet(train_file)
    val_df = spark.read.parquet(val_file)
    print('Finish reading data')

    train_df = train_df.withColumn('inc_count', train_df['count']+1)
    train_df = train_df.withColumn('log_count', F.log(train_df['inc_count']))
    
    print('finish transforming data')
    print(train_df.first())
    print(val_df.first())

    train_df = train_df.select('user_label', 'track_label', 'log_count')
    val_df = val_df.select('user_label', 'track_label', 'count')
    val_grouped = val_df.groupBy('user_label').agg(F.collect_list(F.col('track_label')).alias('track_label'))
    print('finish preparing data') 
    
    val_grouped.cache()
    train_df.cache()
    print('start fitting')
    # ALS for implicit feedback
    als = ALS(maxIter = 5, regParam = 0.01, alpha = 0.1, rank =10, implicitPrefs = True, \
          userCol = 'user_label', itemCol = 'track_label', ratingCol = 'log_count')


    als_model = als.fit(train_df)
    print('Model fitted')
    als_model.save(model_file)
    print('Model Saved')
    predictions = als_model.recommendForAllUsers(100)
    prediction_df = predictions.rdd.map(lambda r: (r.user_label, [i[0] for i in r.recommendations])).toDF()
    prediction_df = prediction_df.selectExpr('_1 as user_label', '_2 as recommendations')

    # Join table
    val_pred = val_grouped.join(prediction_df, 'user_label', 'inner')
    rdd = val_pred.select('recommendations', 'track_label').rdd
    ranking_metrics = RankingMetrics(rdd)
    print('Log: Current log job alpha is : 0.1, current rank is 10, reg is 0.01')
    print('Single model, MAP = %s' % ranking_metrics.meanAveragePrecision)
    #als_model.save(model_file)
    #print('Start tuning parameters')
    # hyperparameter tuning
    #ranks = [20, 40, 60]
    #reg_params = [0.005, 0.01, 0.05]
    #reg_params = [0.5]
    #alphas = [0.20]
    #best_rank = None
    #best_reg_param = None
    #best_alpha = None
    #best_model = None
    #best_map = 0

    #for rank_i, alpha_i, reg_param_i in itertools.product(ranks, alphas, reg_params):

     #   als = ALS(maxIter = 5, regParam = reg_param_i, implicitPrefs = True, alpha = alpha_i,
      #            rank = rank_i, userCol = 'user_label', itemCol = 'track_label', ratingCol = 'count')

       # als_model = als.fit(train_df)
        #predictions = als_model.recommendForAllUsers(100)
        #prediction_df = predictions.rdd.map(lambda r: (r.user_label, [i[0] for i in r.recommendations])).toDF()
        #prediction_df = prediction_df.selectExpr('_1 as user_label', '_2 as recommendations')

        # Join table
        #val_pred = val_grouped.join(prediction_df, 'user_label', 'inner')
        #rdd = val_pred.select('recommendations', 'track_label').rdd
        #ranking_metrics = RankingMetrics(rdd)
        #map_ = ranking_metrics.meanAveragePrecision
        #print('Current rank:', rank_i)
        #print('Current alpha:', alpha_i)
        #print('Current reg:', reg_param_i)
        #print('Current map:', map_)        

        #if map_ > best_map:
        #    best_rank = rank_i
        #    best_reg_param = reg_param_i
        #    best_alpha = alpha_i
        #    best_model = als_model
        #    best_map = map_

    #print('Best rank:', best_rank)
    #print('Best regParam:', best_reg_param)
    #print('Best alpha:', best_alpha)
    #print('Best map:', best_map)

    # save the best model
    #best_model.save(model_file)


if __name__ == '__main__':

    conf = SparkConf()
    conf.set('spark.executor.memory', '8g')
    conf.set('spark.driver.memory', '8g')
    conf.set('spark.default.parallelism', '4')

    spark = SparkSession.builder.config(conf = conf).appName('tuning extension rank 10 a 1 and reg 0.5').getOrCreate()

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    model_file = sys.argv[3]

    main(spark, train_file, val_file, model_file)

