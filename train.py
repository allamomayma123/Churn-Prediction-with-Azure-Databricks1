import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_model(input_file):
    # Charger les données prétraitées
    data = pd.read_csv(input_file)
    
    # Séparer les caractéristiques et la variable cible
    X = data.drop(columns=['left'])
    y = data['left']
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluer le modèle
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Suivre l'expérience avec MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    
    print(f'Model accuracy: {accuracy}')

if __name__ == "__main__":
    train_model('/FileStore/tables/hr_employee_churn_data.csv')
