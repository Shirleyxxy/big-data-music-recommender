#!/usr/bin/env python
# -*- coding: utf-8 -*-

#module load python/gnu/3.6.5
#module load spark/2.4.0
# PYSPARK_PYTHON=$(which python) pyspark

# To run: 
# spark-submit ALS_train.py hdfs:/user/nhl256/cf_test_transformed.parquet hdfs:/user/nhl256/train_als_val_small.model



import sys

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.feature import VectorAssembler


def main(spark, test_file, model_file):
    
    test = spark.read.parquet(test_file)
    test_df = test.select('user_label', 'track_label', 'count')
    
    #model = MatrixFactorizationModel.load(sc, model_file)
    model = ALSModel.load(model_file)
    
    
    # Get the predictions 
    # Generate top 10 movie recommendations for each user
    #predictions = model.recommendForAllUsers(500)
    predictions = model.recommendForAllUsers(10)

    pred_temp = predictions.rdd.flatMapValues(lambda value:value).collect() #pred_temp is a list
    predictions_formated = pred_temp.map(lambda r: ((r[0], r[1][0]), r[1][1]))

    
#     >>> predictions.first()
#     Row(user_label=148, recommendations=[Row(track_label=6, rating=0.8295363187789917), Row(track_label=15, rating=0.6116966009140015), Row(track_label=4, rating=0.5459401607513428), Row(track_label=22, rating=0.5194802284240723), Row(track_label=16, rating=0.49332332611083984), Row(track_label=5, rating=0.4515130817890167), Row(track_label=11, rating=0.43997693061828613), Row(track_label=20, rating=0.40816327929496765), Row(track_label=2, rating=0.39762917160987854), Row(track_label=14, rating=0.39114493131637573)])
    

    
    # Instantiate regression metrics to compare predicted and actual ratings
    ranking_metrics = RankingMetrics(predLabel_rdd)
    
    # MAP
    print("MAP = %s" % ranking_metrics.meanAveragePrecision)
    
    
    # Precision at 4
    print("Precision At 3 = %s" % ranking_metrics.metrics.precisionAt(3))
    
    

if __name__ == '__main__':
    
    # Create the spark session object
    sc = spark.sparkContext
    spark = SparkSession.builder.appName('ALS_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)

