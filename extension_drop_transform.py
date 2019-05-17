#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import monotonically_increasing_id
## Will read in all original data and retransform the data
## select only the counts > 5
## train: 5184103 rows
## val: 5,162 users
## test: 51,551 users

def main(spark, train_file, meta_file, train_new_file):
    train_data = spark.read.parquet(train_file)
    meta_data = spark.read.parquet(meta_file)
    print('Finishe reading all data')

    train_data.createOrReplaceTempView('train_data')
    
    train_data = spark.sql('SELECT * FROM train_data WHERE train_data.count > 2')
    train_idx = train_data.select("*").withColumn("id", monotonically_increasing_id())
    train_idx.createOrReplaceTempView('train_idx')
    train_subset = spark.sql('SELECT * FROM train_idx ORDER BY train_idx.id DESC LIMIT 6000000')
    print('Finish dropping the low counts')
   
    user_indexer = StringIndexer(inputCol = 'user_id', outputCol = 'user_label', handleInvalid = 'skip')
    track_indexer = StringIndexer(inputCol = 'track_id', outputCol = 'track_label', handleInvalid ='skip')
    
    user_model = user_indexer.fit(train_subset)
    track_model = track_indexer.fit(meta_data)
     
    train_data = user_model.transform(train_subset)
    train_data = track_model.transform(train_data)
    print('Finish trasnforming train data') 
   
    print('The rows remained in train:', train_data.first())
    train_data = train_data.repartition(5000, 'user_label')

    train_data.write.parquet(train_new_file)
    print('Finish writing training_data')

if __name__ == '__main__':

    conf = SparkConf()
    conf.set('spark.executor.memory', '8g')
    conf.set('spark.driver.memory', '8g')
    conf.set('spark.default.parallelism', '4')

    # create the spark session object
    spark = SparkSession.builder.config(conf = conf).appName('extension_drop').getOrCreate()

    # paths for the original files
    train_file = sys.argv[1]
    meta_file = sys.argv[2]
    # paths to store the transformed files
    train_new_file = sys.argv[3]

    #Call the main function
    main(spark, train_file, meta_file, train_new_file)

