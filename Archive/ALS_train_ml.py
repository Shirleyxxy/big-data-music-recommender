#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import Row
def main(spark, train_file, model_file):
    train_data = spark.read.parquet(train_file).rdd
    train_data = train_data.sample(False, 0.01)
    # row(user, item(track_label), count)
    train_data = train_data.map(lambda p: Row(user = int(p[4]), item = int(p[5]), rating=float(p[1])))
    print(train_data.take(1))
    #Speed up process
    df = spark.createDataFrame(train_data)
    df.cache()
 
    als = ALS(maxIter = 2, userCol = 'user', \
              itemCol = 'item', ratingCol = 'rating', \
              rank=5, 
              implicitPrefs = True, coldStartStrategy = 'drop')
    
    #pipeline = Pipeline(stages = [als])
    #paramGrid = ParamGridBuilder().addGrid(als.rank, [5, 10]).build()
    #crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,
    #                          evaluator=RegressionEvaluator(predictionCol='prediction', labelCol ='count'),
    #                          numFolds=5)
    #model = crossval.fit(train_data)
    model = als.fit(df)
    
    model.save(model_file)
    #evaluator = RegressionEvaluator(predictionCol ='prediction', labelCol = 'count')
    #print(evaluator.evaluate(predictions))

#    crossval = CrossValidator(estimator= als, 
#                         estimatorParamMaps = paramGrid,
#                         evaluator = RegressionEvaluator(metricName = 'rmse',labelCol = 'count',predictionCol = 'prediction'),
#                         numFolds = 3)
    
#    model = crossval.fit(train_data)
#   best_model = model.bestModel
#   best_model.save(model_file)

if __name__ == '__main__':
    spark = SparkSession.builder.appName('ALS_train').getOrCreate()
    #Get the file path
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    
    main(spark, train_file, model_file) 
  
