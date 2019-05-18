#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def main(spark, train_file, val_file, test_file, train_output_file, 
         val_output_file, test_output_file):
    train_data = spark.read.parquet(train_file)
    val_data = spark.read.parquet(val_file)
    test_data = spark.read.parquet(test_file)
    
    train_subset = train_data.sample(False, 0.00003)
    val_subset = val_data.sample(False, 0.01)
    test_subset = test_data.sample(False, 0.001) 
    
    train_subset.createOrReplaceTempView('train_subset')
    val_subset.createOrReplaceTempView('val_subset')
    test_subset.createOrReplaceTempView('test_subset')

    spark.sql('SELECT COUNT(*) FROM train_subset').show()
    spark.sql('SELECT COUNT(*) FROM val_subset').show()
    spark.sql('SELECT COUNT(*) FROM test_subset').show()
    
    train_subset = train_subset.drop('__index_level_0__')
    val_subset = val_subset.drop('__index_level_0__')
    test_subset = test_subset.drop('__index_level_0__')
 
    print(train_subset.take(1))
    print(val_subset.take(1))
    print(test_subset.take(1))

    user_indexer = StringIndexer(inputCol = 'user_id', outputCol = 'user_label', 
				handleInvalid = 'skip')
    track_indexer = StringIndexer(inputCol = 'track_id', outputCol = 'track_label', 
				handleInvalid = 'skip')
    
    indexer_pipeline = Pipeline(stages = [user_indexer, track_indexer])
    
    train_model = indexer_pipeline.fit(train_subset)
    train_subset = train_model.transform(train_subset)
   
    val_model = indexer_pipeline.fit(val_subset)
    val_subset = val_model.transform(val_subset)

    test_model = indexer_pipeline.fit(test_subset)
    test_subset = test_model.transform(test_subset)
    print(train_subset.take(1))
    print(val_subset.take(1))
    print(test_subset.take(1))

    train_subset.write.parquet(train_output_file)
    val_subset.write.parquet(val_output_file)
    test_subset.write.parquet(test_output_file)


if __name__ == '__main__':
    spark = SparkSession.builder.appName('subset_data').getOrCreate()
    #Get the file path
    train_file  = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]
    
    train_output_file = sys.argv[4]
    val_output_file = sys.argv[5]
    test_output_file = sys.argv[6]

    #Call the main function
    main(spark, train_file, val_file, test_file, train_output_file, 
         val_output_file, test_output_file)
