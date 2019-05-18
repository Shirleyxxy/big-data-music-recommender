#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS
#from pyspark.ml.evaluation import RegressionEvaluator
from math import sqrt
from operator import add


def main(spark, train_file, val_file, model_file):

    train_df = spark.read.parquet(train_file)
    #train_df = spark.read.parquet('hdfs:/user/xx852/cf_train_small.parquet')
    val_df = spark.read.parquet(val_file)
    #val_df = spark.read.parquet('hdfs:/user/xx852/cf_val_small.parquet')
    train_df = train_df.select('user_label', 'track_label', 'count')
    val_df = val_df.select('user_label', 'track_label', 'count')

    train_rdd = train_df.rdd.map(lambda x: (int(x[0]), int(x[1]), int(x[2])))

    # ALS for implicit feedback
    model = ALS.trainImplicit(train_rdd, rank = 5, iterations = 5)

    # evaluation
    pred_input = val_df.rdd.map(lambda x: (int(x[0]), int(x[1])))
    val_rdd = val_df.rdd.map(lambda x: ((int(x[0]), int(x[1])), int(x[2])))
    pred = model.predictAll(pred_input).map(lambda x: ((x[0], x[1]), x[2]))
    predAndCount = val_rdd.join(pred).values()

    mse = predAndCount.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / predAndCount.count()
    rmse = sqrt(mse)
    print('rmse before tuning:', rmse) # the value 5.22 corresponds to the rmse from RegressionEvaluator

    #hyperparameter tuning
    ranks = [10, 25, 50, 100]
    reg_params = [0.01, 0.05, 0.10, 0.50, 1.0]
    alphas = [0.25, 0.50, 0.75, 1.00]
    best_rank = None
    best_reg_param = None
    best_alpha = None
    best_model = None
    best_rmse = 100000 # set an arbitrarily large number

    for rank, reg_param, alpha in itertools.product(ranks, reg_params, alphas):

        model = ALS.trainImplicit(train_rdd, iterations = 5, rank = rank,
                                  lambda_ = reg_param, alpha = alpha)

        pred = model.predictAll(pred_input).map(lambda x: ((x[0], x[1]), x[2]))
        predAndCount = val_rdd.join(pred).values()
        mse = predAndCount.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / predAndCount.count()
        rmse = sqrt(mse)

        if rmse < best_rmse:
            best_rank = rank
            best_reg_param = reg_param
            best_alpha = alpha
            best_model = model
            best_rmse = rmse

    print('Best rank:', best_rank)
    print('Best reg_param:', best_reg_param)
    print('Best alpha:', best_alpha)
    print('Best rmse:', best_rmse)

    # save the best model
    best_model.save(spark.sparkContext, path = model_file)


if __name__ == '__main__':

    spark = SparkSession.builder.appName('modeling').getOrCreate()

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    model_file = sys.argv[3]

    main(spark, train_file, val_file, model_file)

