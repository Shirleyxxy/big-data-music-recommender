#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark.mllib.evaluation import RankingMetrics

def main(spark, model_file, test_file):

    model = MatrixFactorizationModel.load(sc, model_file)
    #model = MatrixFactorizationModel.load(sc, 'hdfs:/user/xx852/als_model')
    test_df = spark.read.parquet(test_file)
    #test_df = spark.read.parquet('hdfs:/user/xx852/cf_test_small.parquet')
    test_df = test_df.select('user_label', 'track_label', 'count')

    #predictions = model.recommendProductsForUsers(500)
    predictions = model.recommendProductsForUsers(2)
    prediction_flat = predictions.flatMap(lambda p: p[1])
    prediction_df = prediction_flat.toDF()
    intersections = prediction_df.join(test_df, (prediction_df.product == test_df.track_label)&
                                      (prediction_df.user == test_df.user_label), how = 'inner')
    predLabel = intersections.select('rating', 'count')
    #predLabel_rdd = predLabel.rdd.map(lambda x: Row(x[0], x[1]))
    #metrics = RankingMetrics(predLabel_rdd)
    #print(metrics.meanAveragePrecision)

    from pyspark.sql import Window
    import pyspark.sql.functions as psf
    w_rating = Window.orderBy(psf.desc('rating'))
    w_count = Window.orderBy(psf.desc('count'))
    predLabel = predLabel.withColumn('rating_rank', \
                psf.dense_rank().over(w_rating)).withColumn('count_rank', \
                psf.dense_rank().over(w_count))

    predLabel = predLabel.select('rating_rank', 'count_rank')
    #predLabel_rdd = predLabel.rdd.map(lambda x: Row(x[0], x[1]))
    predLabel_rdd = predLabel.rdd
    metrics = RankingMetrics(predLabel_rdd)









if __name__ == '__main__':

    spark = SparkSession.builder.appName('evaluation').getOrCreate()
    sc = spark.sparkContext

    model_file = sys.argv[1]
    test_file = sys.argv[2]

    main(spark, model_file, test_file)

