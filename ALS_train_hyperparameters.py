#!/usr/bin/env python
# -*- coding: utf-8 -*-

#module load python/gnu/3.6.5
#module load spark/2.4.0
# PYSPARK_PYTHON=$(which python) pyspark

# To run: 
# spark-submit ALS_train.py hdfs:/user/nhl256/cf_val_transformed.parquet hdfs:/user/nhl256/train_als_val_small.model


import sys

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
# from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating



def main(spark, train_file, val_file, model_file):
    
    train = spark.read.parquet(train_file)
    val = spark.read.parquet(val_file)
    
    # A Ratings object is made up of (user, item, rating)
    train_ratings = train.rdd.map(lambda x: Rating(int(x[4]), int(x[5]), float(x[1])))
    val_ratings = val.rdd.map(lambda x: Rating(int(x[4]), int(x[5]), float(x[1])))
    
    ####### Archived work using MLLiB ###########
    
    # Try with sample of 10% of data first
    df = df.sample(False, 0.1)
    
    # A Ratings object is made up of (user, item, rating)
    ratings = df.rdd.map(lambda x: Rating(int(x[4]), int(x[5]), float(x[1])))
    
    #Setting up the parameters for ALS
    rank = 5 # Latent Factors to be made
    numIterations = 10 # Times to repeat process
    #Create the model on the training data
    model = ALS.trainImplicit(ratings, rank, numIterations)
    
    model.save(spark.SparkContext, model_file)

if __name__ == '__main__':
    
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS_train').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    val_file = sys.argv[2]

    # And the location to store the trained model
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, val_file, model_file)

