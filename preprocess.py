import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_file, output_file):
    # Charger les données
    data = pd.read_csv(input_file)
    
    # Supprimer la colonne empid car elle n'est pas utile pour le modèle
    data = data.drop(columns=['empid'])
    
    # Encoder les variables catégorielles
    le = LabelEncoder()
    data['salary'] = le.fit_transform(data['salary'])
    
    # Normaliser les données
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    # Sauvegarder les données prétraitées
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_data('/FileStore/tables/hr_employee_churn_data.csv', '/FileStore/tables/hr_employee_churn_data.csv')
