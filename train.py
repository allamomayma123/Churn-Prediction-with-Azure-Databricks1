import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_model(input_file):
    # Load the preprocessed data
    data = pd.read_csv(input_file)
    
    # Separate features and target variable
    X = data.drop(columns=['left'])
    y = data['left']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Track the experiment with MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    
    print(f'Model accuracy: {accuracy}')

if __name__ == "__main__":
    train_model('/processed_hr_employee_churn_data.csv')
