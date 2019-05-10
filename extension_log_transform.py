#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# This script transform the dataset 
# spark-submit --driver-memory 16g --executor-memory 16g extension_perform.py 
# hdfs:/user/ht1162/train_new.parquet hdfs:/user/ht1162/val_new.parquet 
# hdfs:/user/ht1162/test_new.parquet hdfs:/user/ht1162/train_log.parquet
# hdfs:/user/ht1162/val_log.parquet hdfs:/user/ht1162/test_log.parquet 

def main(spark, train_file, val_file, test_file, train_log_file, val_log_file, test_log_file):
    train_data = spark.read.parquet(train_file)
    val_data = spark.read.parquet(val_file)
    test_data = spark.read.parque(test_file)
    print('Finishe reading all data')

    train_log = train_data.withColumn('log_count', F.log('count'))
    val_log = val_data.withColumn('log_count', F.log('count'))
    test_log = test_data.withColumn('log_count', F.log('count'))
    print('Finish adding a column')

    train_log = train_log.repartition(5000, 'user_label')
    val_log = val_log.repartition('user_label')
    test_log = test_log.repartition('user_label')    

    train_log.write.parquet(train_log_file)
    print('Finish writing training_data')
    val_log.write.parquet(val_log_file)
    print('Finish writing val_data')
    test_log.write.parquet(test_log_file)
    print('Finish writing test_data')


if __name__ == '__main__':

    conf = SparkConf()
    conf.set('spark.executor.memory', '16g')
    conf.set('spark.driver.memory', '16g')
    conf.set('spark.default.parallelism', '4')

    # create the spark session object
    spark = SparkSession.builder.config(conf = conf).appName('extension_log').getOrCreate()

    # paths for the original files
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    # paths to store the transformed files
    train_log_file = sys.argv[4]
    val_log_file = sys.argv[5]
    test_log_file = sys.argv[6]

    #Call the main function
    main(spark, train_file, val_file, test_file, train_log_file,
         val_log_file, test_log_file)
