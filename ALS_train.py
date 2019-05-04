#!/usr/bin/env python
# -*- coding: utf-8 -*-


# To run: 
# spark-submit ALS_train.py hdfs:/user/nhl256/cf_val_transformed.parquet hdfs:/user/nhl256/val_als.model


import sys

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.ml.feature import VectorAssembler


def main(spark, data_file, model_file):
    
    df = spark.read.parquet(data_file)
    
    # Try with sample of 10% of data first
    #df = df.sample(False, 0.1)
    
    # A Ratings object is made up of (user, item, rating)
    #columns = ['user_label', 'track_label', 'count']
    #assembler = VectorAssembler( inputCols=columns, outputCol="features")
    #training = assembler.transform(df)
    
    # A Ratings object is made up of (user, item, rating)
    #df = df.rdd.map(lambda l: l.split('\t'))
    ratings = df.rdd.map(lambda x: Rating(int(x[4]), int(x[5]), float(x[1])))
    
    #Setting up the parameters for ALS
    rank = 5 # Latent Factors to be made
    numIterations = 10 # Times to repeat process
    #Create the model on the training data
    model = ALS.trainImplicit(ratings, rank, numIterations)
    
    model.write().overwrite().save(model_file)

if __name__ == '__main__':
    
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)

