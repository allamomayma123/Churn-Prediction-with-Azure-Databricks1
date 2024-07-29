from pyspark.sql import SparkSession
import pandas as pd
import numpy as np 

from pyspark.sql.functions import mean, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

def preprocess_data(input_path, output_path):
    # Initialize Spark session
    spark = SparkSession.builder.appName("HR Employee Churn").getOrCreate()

    # Load the data into a Spark DataFrame
    df_spark = spark.read.format("csv") \
      .option("inferSchema", "true") \
      .option("header", "true") \
      .option("sep", ",") \
      .load(input_path)

    # Handle missing values
    mean_satisfaction_level = df_spark.select(mean(col('satisfaction_level'))).collect()[0][0]
    df_spark = df_spark.na.fill({'satisfaction_level': mean_satisfaction_level})

    # Encode categorical variables using StringIndexer and OneHotEncoder
    indexer = StringIndexer(inputCol="salary", outputCol="salaryIndex")
    df_spark = indexer.fit(df_spark).transform(df_spark)

    encoder = OneHotEncoder(inputCols=["salaryIndex"], outputCols=["salaryVec"])
    df_spark = encoder.fit(df_spark).transform(df_spark)

    # Use VectorAssembler to combine all feature columns into a single vector column
    assembler = VectorAssembler(inputCols=['satisfaction_level', 'last_evaluation', 'number_project', 
                                           'average_montly_hours', 'time_spend_company', 'Work_accident', 
                                           'promotion_last_5years', 'salaryVec'], outputCol="features")
    df_spark = assembler.transform(df_spark)

    # Drop the intermediate columns
    df_spark = df_spark.drop("salary", "salaryIndex", "salaryVec")

    # Convert Spark DataFrame to Pandas DataFrame
    df_pandas = df_spark.select("empid", "features", "left").toPandas()

    # Function to extract features from the vector column
    def extract_features(features_vector):
        return features_vector.toArray()

    # Apply the function to extract features
    features_array = np.array(df_pandas['features'].apply(lambda x: x.toArray()).tolist())

    # Define feature names
    feature_names = ['satisfaction_level', 'last_evaluation', 'number_project', 
                     'average_montly_hours', 'time_spend_company', 'Work_accident', 
                     'promotion_last_5years', 'salaryVec_0', 'salaryVec_1']

    # Create a DataFrame from the features array
    features_df = pd.DataFrame(features_array, columns=feature_names)

    # Combine the features DataFrame with the target and ID columns
    df_pandas = pd.concat([df_pandas[['empid', 'left']], features_df], axis=1)

    # Save the preprocessed DataFrame to a CSV file
    df_pandas.to_csv(output_path, index=False)

    print(f"Preprocessed data saved to {output_path}")

# Call the preprocess_data function with the correct DBFS paths
preprocess_data('/FileStore/tables/hr_employee_churn_data.csv', '/processed_hr_employee_churn_data.csv')
