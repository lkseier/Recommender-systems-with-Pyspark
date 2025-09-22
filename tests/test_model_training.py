import unittest
from pyspark.sql import SparkSession
from src.recommender_utils import drop_missing, cast_columns, train_als

class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master('local[1]').appName('TestRecommender').getOrCreate()
        # Create a small sample DataFrame
        cls.data = [
            (1, 10, 4.0),
            (1, 20, 5.0),
            (2, 10, 3.0),
            (2, 30, 2.0),
            (3, 20, 1.0)
        ]
        cls.columns = ['userId', 'movieId', 'rating']
        cls.df = cls.spark.createDataFrame(cls.data, cls.columns)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_drop_missing(self):
        df_clean = drop_missing(self.df)
        self.assertEqual(df_clean.count(), 5)

    def test_cast_columns(self):
        df_casted = cast_columns(self.df)
        self.assertEqual(df_casted.schema['userId'].dataType.typeName(), 'integer')
        self.assertEqual(df_casted.schema['movieId'].dataType.typeName(), 'integer')
        self.assertEqual(df_casted.schema['rating'].dataType.typeName(), 'float')

    def test_train_als(self):
        model = train_als(self.df)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'transform'))

if __name__ == '__main__':
    unittest.main()
