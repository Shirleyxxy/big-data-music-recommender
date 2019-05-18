#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id

def main(spark, train_file, val_file, test_file):
    train_data = spark.read.parquet(train_file)
    val_data = spark.read.parquet(val_file)
    test_data = spark.read.parquet(test_file)
    
   
    train_data.createOrReplaceTempView('train_data')
    val_data.createOrReplaceTempView('val_data')
    test_data.createOrReplaceTempView('test_data')
    print('The original data set has lines: ', train_data.count()) 
    train_data = spark.sql('SELECT * FROM train_data WHERE train_data.count > 1')
#    train_idx = train_data.select("*").withColumn("id", monotonically_increasing_id())
#    train_idx.createOrReplaceTempView('train_idx')
#    train_subset = spark.sql('SELECT * FROM train_idx ORDER BY train_idx.id DESC LIMIT 4500000') 
   
    print('The number distribution for train data is :', train_data.count())
    
    print('The intersections of users')
    #spark.sql('SELECT COUNT(DISTINCT train_data.user_id) FROM train_data').show()
    spark.sql('SELECT COUNT(DISTINCT train_data.user_id) FROM train_data INNER JOIN val_data ON train_data.user_id = val_data.user_id').show()
    spark.sql('SELECT COUNT(DISTINCT val_data.user_id) FROM val_data').show()

    spark.sql('SELECT COUNT(DISTINCT train_data.user_id) FROM train_data INNER JOIN test_data ON train_data.user_id = test_data.user_id').show()
    spark.sql('SELECT COUNT(DISTINCT test_data.user_id) FROM test_data').show()

if __name__ == '__main__':

    conf = SparkConf()
    conf.set('spark.executor.memory', '8g')
    conf.set('spark.driver.memory', '8g')
    conf.set('spark.default.parallelism', '4')

    # create the spark session object
    spark = SparkSession.builder.config(conf = conf).appName('drop_check').getOrCreate()

    # paths for the original files
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    #Call the main function
    main(spark, train_file, val_file, test_file)

