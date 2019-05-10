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
    #train_df = spark.read.parquet('hdfs:/user/xx852/cf_train_small.parquet')
    val_df = spark.read.parquet(val_file)
    #val_df = spark.read.parquet('hdfs:/user/xx852/cf_val_small.parquet')
    train_df = train_df.select('user_label', 'track_label', 'count')
    val_df = val_df.select('user_label', 'track_label', 'count')
    val_grouped = val_df.groupBy('user_label').agg(F.collect_list(F.col('track_label')).alias('track_label'))

    # ALS for implicit feedback
    als = ALS(maxIter = 5, regParam = 0.01, implicitPrefs = True, \
          userCol = 'user_label', itemCol = 'track_label', ratingCol = 'count')

    als_model = als.fit(train_df)
    predictions = als_model.recommendForAllUsers(10)
    prediction_df = predictions.rdd.map(lambda r: (r.user_label, [i[0] for i in r.recommendations])).toDF()
    prediction_df = prediction_df.selectExpr('_1 as user_label', '_2 as recommendations')

    # Join table
    val_pred = val_grouped.join(prediction_df, 'user_label', 'inner')
    rdd = val_pred.select('recommendations', 'track_label').rdd
    ranking_metrics = RankingMetrics(rdd)
    print('Before tuning, MAP = %s' % ranking_metrics.meanAveragePrecision)


    # hyperparameter tuning
    #ranks = [10, 20, 40, 60]
    #reg_params = [0.005, 0.01, 0.05]
    ranks = [20]
    reg_params = [0.01]
    alphas = [0.10, 0.20, 0.40]
    best_rank = None
    best_reg_param = None
    best_alpha = None
    best_model = None
    best_map = 0

    for rank_i, alpha_i, reg_param_i in itertools.product(ranks, alphas, reg_params):

        als = ALS(maxIter = 5, regParam = reg_param_i, implicitPrefs = True, alpha = alpha_i,
                  rank = rank_i, userCol = 'user_label', itemCol = 'track_label', ratingCol = 'count')

        als_model = als.fit(train_df)
        predictions = als_model.recommendForAllUsers(100)
        prediction_df = predictions.rdd.map(lambda r: (r.user_label, [i[0] for i in r.recommendations])).toDF()
        prediction_df = prediction_df.selectExpr('_1 as user_label', '_2 as recommendations')

        # Join table
        val_pred = val_grouped.join(prediction_df, 'user_label', 'inner')
        rdd = val_pred.select('recommendations', 'track_label').rdd
        ranking_metrics = RankingMetrics(rdd)
        map_ = ranking_metrics.meanAveragePrecision
        print('Current rank:', rank_i)
        print('Current alpha:', alpha_i)
        print('Current reg:', reg_param_i)
        print('Current map:', map_)

        if map_ > best_map:
            best_rank = rank_i
            best_reg_param = reg_param_i
            best_alpha = alpha_i
            best_model = als_model
            best_map = map_

    print('Best rank:', best_rank)
    print('Best regParam:', best_reg_param)
    print('Best alpha:', best_alpha)
    print('Best map:', best_map)

    # save the best model
    best_model.save(model_file)


if __name__ == '__main__':

    conf = SparkConf()
    conf.set('spark.executor.memory', '16g')
    conf.set('spark.driver.memory', '16g')
    conf.set('spark.default.parallelism', '4')

    spark = SparkSession.builder.config(conf = conf).appName('modeling and tuning').getOrCreate()

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    model_file = sys.argv[3]

    main(spark, train_file, val_file, model_file)

