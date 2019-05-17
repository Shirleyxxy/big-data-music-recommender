#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F


def main(spark, val_file, model_file):
    model = ALSModel.load(model_file)
    print('finish loading models')
    val_df = spark.read.parquet(val_file)
    val_df = val_df.select('user_label', 'track_label')
    val_grouped = val_df.groupBy('user_label').agg(F.collect_list(F.col('track_label')).alias('track_label'))
    print('Finish preparing test data')
    val_grouped.cache()

    predictions = model.recommendForAllUsers(500)
    print('finish making predictions')
    prediction_df = predictions.rdd.map(lambda r: (r.user_label, [i[0] for i in r.recommendations])).toDF()
    prediction_df = prediction_df.selectExpr("_1 as user_label", "_2 as recommendations")

    # Join table
    val_pred = val_grouped.join(prediction_df, "user_label", "inner")
    print('finish joining data')
    # Instantiate regression metrics to compare predicted and actual ratings
    rdd = val_pred.select('recommendations', 'track_label').rdd
    print('final steps')
    ranking_metrics = RankingMetrics(rdd)

    # MAP
    print("MAP = %s" % ranking_metrics.meanAveragePrecision)


if __name__ == '__main__':
    conf = SparkConf()
    conf.set('spark.executor.memory', '8g')
    conf.set('spark.driver.memory', '8g')
    conf.set('spark.default.parallelism', '4')
    spark = SparkSession.builder.appName('evaluation_test').getOrCreate()

    val_file = sys.argv[1]
    model_file = sys.argv[2]

    main(spark,val_file, model_file)

