#!/usr/bin/env python
# -*- coding: utf-8 -*-

#module load python/gnu/3.6.5
#module load spark/2.4.0
# PYSPARK_PYTHON=$(which python) pyspark

# To run: 
# spark-submit ALS_train.py hdfs:/user/nhl256/cf_test_transformed.parquet hdfs:/user/nhl256/val_als.model


import sys

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.ml.feature import VectorAssembler


def main(spark, data_file, model_file):
    
    df = spark.read.parquet(data_file)
    
    model = MatrixFactorizationModel.load(model_file)
    
    # Transform data_file - A Ratings object is made up of (user, item, rating)
    predD_input = df.rdd.map(lambda x: Rating(int(x[4]), int(x[5])))
    pred = model.predictAll(pred_input) 
    
    #Organize the data to make (user, product) the key)
    true_reorg = df.rdd.map(lambda x:((x[4], x[5]), x[1]))
    pred_reorg = pred.map(lambda x:((x[4],x[5]), x[2]))

    #Do the actual join
    true_pred = true_reorg.join(pred_reorg)

    #Need to be able to square root the Mean-Squared Error
    from math import sqrt

    MSE = true_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    RMSE = sqrt(MSE)#Results in 0.7629908117414474
    

if __name__ == '__main__':
    
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)

