#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# This script transform the dataset 
# spark-submit --driver-memory 16g --executor-memory 16g extension_perform.py 
# hdfs:/user/ht1162/train_new.parquet hdfs:/user/ht1162/train_log.parquet

def main(spark, train_file, train_log_file):
    train_data = spark.read.parquet(train_file)
    print('Finishe reading all data')
    
    #Log transform the counts for training set
    train_data = train_data.withColumn('inc_count', train_data['count']+1)
    train_log = train_data.withColumn('log_count', F.log(train_data['inc_count']))
    print('Finish adding a column')

    train_log = train_log.repartition(5000, 'user_label')

    train_log.write.parquet(train_log_file)
    print('Finish writing training_data')


if __name__ == '__main__':

    conf = SparkConf()
    conf.set('spark.executor.memory', '16g')
    conf.set('spark.driver.memory', '16g')
    conf.set('spark.default.parallelism', '4')

    # create the spark session object
    spark = SparkSession.builder.config(conf = conf).appName('extension_log').getOrCreate()

    # paths for the original files
    train_file = sys.argv[1]

    # paths to store the transformed files
    train_log_file = sys.argv[2]

    #Call the main function
    main(spark, train_file, train_log_file)
