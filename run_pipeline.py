from pyspark.sql import SparkSession
from src.recommender_utils import (
    drop_missing, cast_columns, basic_stats, train_als, evaluate_model, get_data_path
)

def main():
    # Start Spark session
    spark = SparkSession.builder \
        .appName("movie-als-training") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .getOrCreate()

    # Load data
    data_path = get_data_path('ratings.csv')
    print(f'Loading data from: {data_path}')
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Data exploration
    print('Data exploration:')
    basic_stats(df)

    # Preprocessing
    print('Preprocessing...')
    df = drop_missing(df)
    df = cast_columns(df)

    # Split data
    print('Splitting data...')
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # Train model

    print('Training ALS model...')
    model = train_als(train)

    # Ensure als_model directory exists
    import os
    model_dir = 'als_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model with overwrite if exists
    print(f'Saving model to {model_dir}/ (overwrite if exists)')
    model.write().overwrite().save(model_dir)

    # Evaluate
    print('Evaluating model...')
    evaluate_model(model, test)

    # Stop Spark session
    spark.stop()

if __name__ == '__main__':
    main()
