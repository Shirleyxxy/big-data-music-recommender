#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


def main(spark, train_file, val_file, test_file, train_output_file,
         val_output_file, test_output_file):

    train_data = spark.read.parquet(train_file)
    val_data = spark.read.parquet(val_file)
    test_data = spark.read.parquet(test_file)

    # print out the information about the original datasets
    print('The number of records in the train set: ', train_data.count())
    print('The number of unique users in the train set: ', train_data.select('user_id').distinct().count())

    print('The number of records in the validation set: ', val_data.count())
    print('The number of unique users in the validation set: ', val_data.select('user_id').distinct().count())

    print('The number of records in the test set: ', test_data.count())
    print('The number of unique users in the test set: ', test_data.select('user_id').distinct().count())

    # drop the unnecessary columns
    train_data = train_data.drop('__index_level_0__')
    val_data = val_data.drop('__index_level_0__')
    test_data = test_data.drop('__index_level_0__')

    # transform the user and item identifiers
    user_indexer = StringIndexer(inputCol = 'user_id', outputCol = 'user_label', handleInvalid = 'skip')
    track_indexer = StringIndexer(inputCol = 'track_id', outputCol = 'track_label', handleInvalid = 'skip')

    indexer_pipeline = Pipeline(stages = [user_indexer, track_indexer])

    train_model = indexer_pipeline.fit(train_data)
    train_data = train_model.transform(train_data)
    val_data = train_model.transform(val_data)
    test_data = train_model.transform(test_data)

    # repartition the data frame prior to writing (fix for java.lang.OutOfMemoryError)

    #print('Number of partitions for the training data:', train_data.rdd.getNumPartitions())
    train_data = train_data.repartition(5000, 'user_label')
    val_data = val_data.repartition('user_label')
    test_data = test_data.repartition('user_label')

    # write the transformed data files
    train_data.write.parquet(train_output_file)
    val_data.write.parquet(val_output_file)
    test_data.write.parquet(test_output_file)


if __name__ == '__main__':

    conf = SparkConf()
    conf.set('spark.executor.memory', '16g')
    conf.set('spark.driver.memory', '16g')
    conf.set('spark.default.parallelism', '4')

    # create the spark session object
    spark = SparkSession.builder.config(conf = conf).appName('preprocessing').getOrCreate()

    # paths for the original files
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    # paths to store the transformed files
    train_output_file = sys.argv[4]
    val_output_file = sys.argv[5]
    test_output_file = sys.argv[6]


    main(spark, train_file, val_file, test_file, train_output_file,
         val_output_file, test_output_file)

