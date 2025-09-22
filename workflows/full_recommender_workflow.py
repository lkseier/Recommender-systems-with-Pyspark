# Full Recommender System Workflow
# This script covers data exploration, preprocessing, modeling, and evaluation in one place.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Start Spark session
spark = SparkSession.builder.appName('RecommenderFullWorkflow').getOrCreate()

# 1. Data Exploration
print('Loading data...')
df = spark.read.csv('../data/ratings.csv', header=True, inferSchema=True)
df.show(5)
df.describe().show()

# 2. Preprocessing & Feature Engineering
print('Preprocessing data...')
df = df.dropna()
df = df.withColumn('userId', df['userId'].cast(IntegerType()))
df = df.withColumn('movieId', df['movieId'].cast(IntegerType()))
df = df.withColumn('rating', df['rating'].cast(FloatType()))
df.printSchema()

# 3. Modeling
print('Splitting data and training ALS model...')
train, test = df.randomSplit([0.8, 0.2], seed=42)
als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')
model = als.fit(train)

# 4. Evaluation
print('Evaluating model...')
predictions = model.transform(test)
predictions.show(5)
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
print(f'Root-mean-square error = {rmse}')

# Stop Spark session
spark.stop()
