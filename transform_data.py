#!/usr/bin/env python
# -*- coding: utf-8 -*-
# module load python/gnu/3.6.5
# module load spark/2.4.0

import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import monotonically_increasing_id

# train_file = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
# val_file = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
# test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
# train_output_file = 'hdfs:/user/nhl256/cf_train_transformed_v4.parquet'
# val_output_file = 'hdfs:/user/nhl256/cf_val_transformed_v1.parquet'
# test_output_file = 'hdfs:/user/nhl256/cf_test_transformed_v1.parquet'

## Command to run: spark-submit --driver-memory 16g --executor-memory 16g transform_data.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet hdfs:/user/bm106/pub/project/cf_test.parquet hdfs:/user/nhl256/cf_train_transformed_v8.parquet hdfs:/user/nhl256/cf_val_transformed_v1.parquet hdfs:/user/nhl256/cf_test_transformed_v1.parquet

def main(spark, train_file, val_file, test_file, meta_file, train_output_file,
         val_output_file, test_output_file):

    train_data = spark.read.parquet(train_file)
    val_data = spark.read.parquet(val_file)
    test_data = spark.read.parquet(test_file)
    meta_data = spark.read.parquet(meta_file)

    # Drop '__index_level_0__' columns
    train_data = train_data.drop('__index_level_0__')
    val_data = val_data.drop('__index_level_0__')
    test_data = test_data.drop('__index_level_0__')

    train_idx = train_data.select("*").withColumn("id", monotonically_increasing_id())
    train_idx.creatOrReplaceTempView('train_idx')
    train_subset = spark.sql('SELECT * FROM train_idx ORDER BY train_idx.id DESC LIMIT 6000000')

    user_indexer = StringIndexer(inputCol = 'user_id', outputCol = 'user_label', handleInvalid = 'skip')
    track_indexer = StringIndexer(inputCol = 'track_id', outputCol = 'track_label', handleInvalid = 'skip')

    #indexer_pipeline = Pipeline(stages = [user_indexer, track_indexer])

    # train subset will contain all users from val and test
    user_model = user_indexer.fit(train_subset)
    # meta data track file will contain all track infos
    track_model = track_indexer.fit(meta_data)

    train_data = user_model.transform(train_subset)
    train_data = track_model.transform(train_data)

    val_data = user_model.transform(val_data)
    val_data = track_model.transform(val_data)

    test_data = user_model.transform(test_data)
    test_data = track_model.transform(test_data)

    # Make sure that train, val, and test have been transformed
    print(train_data.take(1))
    print(val_data.take(1))
    print(test_data.take(1))

    # repartition the data frame prior to writing (fix for java.lang.OutOfMemoryError)

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
    meta_file = sys.argv[4]

    # paths to store the transformed files
    train_output_file = sys.argv[5]
    val_output_file = sys.argv[6]
    test_output_file = sys.argv[7]


    #Call the main function
    main(spark, train_file, val_file, test_file, meta_file, train_output_file,
         val_output_file, test_output_file)
