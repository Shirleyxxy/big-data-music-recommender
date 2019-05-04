#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# train_file = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
# val_file = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
# test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
# train_output_file = 'hdfs:/user/nhl256/cf_train_transformed_v2.parquet'
# val_output_file = 'hdfs:/user/nhl256/cf_val_transformed.parquet'
# test_output_file = 'hdfs:/user/nhl256/cf_test_transformed.parquet'

## Command to run: spark-submit transform_data.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet hdfs:/user/bm106/pub/project/cf_test.parquet hdfs:/user/nhl256/cf_train_transformed.parquet hdfs:/user/nhl256/cf_val_transformed.parquet hdfs:/user/nhl256/cf_test_transformed.parquet

def main(spark, train_file, val_file, test_file, train_output_file, 
         val_output_file, test_output_file):
    
    train_data = spark.read.parquet(train_file)
    val_data = spark.read.parquet(val_file)
    test_data = spark.read.parquet(test_file)
    
    
    user_indexer = StringIndexer(inputCol = 'user_id', outputCol = 'user_label', handleInvalid = 'skip')
    track_indexer = StringIndexer(inputCol = 'track_id', outputCol = 'track_label', handleInvalid = 'skip')
    
    indexer_pipeline = Pipeline(stages = [user_indexer, track_indexer])
    
    train_model = indexer_pipeline.fit(train_data)
    train_data = train_model.transform(train_data)
   
    val_model = indexer_pipeline.fit(val_data)
    val_data = val_model.transform(val_data)

    test_model = indexer_pipeline.fit(test_data)
    test_data = test_model.transform(test_data)
    
    print(train_data.take(1))
    print(val_data.take(1))
    print(test_data.take(1))

    train_data.write.parquet(train_output_file)
    val_data.write.parquet(val_output_file)
    test_data.write.parquet(test_output_file)


if __name__ == '__main__':
    
    # Create the spark session object
    spark = SparkSession.builder.appName('transform_data').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    # And the location to store the transformed file
    train_output_file = sys.argv[4]
    val_output_file = sys.argv[5]
    test_output_file = sys.argv[6]



    #Call the main function
    main(spark, train_file, val_file, test_file, train_output_file, 
         val_output_file, test_output_file)
