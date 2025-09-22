import os
from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator

# --- Data Preprocessing ---
def drop_missing(df: DataFrame) -> DataFrame:
    """Drop rows with missing values."""
    return df.dropna()

def cast_columns(df: DataFrame) -> DataFrame:
    """Cast userId and movieId to int, rating to float."""
    from pyspark.sql.types import IntegerType, FloatType
    df = df.withColumn('userId', df['userId'].cast(IntegerType()))
    df = df.withColumn('movieId', df['movieId'].cast(IntegerType()))
    df = df.withColumn('rating', df['rating'].cast(FloatType()))
    return df

# --- Feature Engineering ---
def basic_stats(df: DataFrame):
    """Show basic statistics and schema."""
    df.show(5)
    df.describe().show()
    df.printSchema()

# --- Model Training ---
def train_als(train: DataFrame) -> ALSModel:
    """Train ALS model on training data."""
    als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')
    model = als.fit(train)
    return model

# --- Evaluation ---
def evaluate_model(model: ALSModel, test: DataFrame) -> float:
    """Evaluate ALS model using RMSE."""
    predictions = model.transform(test)
    predictions.show(5)
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
    rmse = evaluator.evaluate(predictions)
    print(f'Root-mean-square error = {rmse}')
    return rmse

# --- Utils ---
def get_data_path(filename: str) -> str:
    """Get the absolute path to a data file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', filename)
