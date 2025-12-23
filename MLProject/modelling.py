import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- SETUP DAGSHUB ---
dagshub.init(repo_owner='Tikleboy', repo_name='Eksperimen_SML_StanlyLopez', mlflow=True)
mlflow.set_experiment("CI_Pipeline_Experiment")

def train_model():
    print("Loading data...")
    # Path relatif: script ada di folder MLProject, data ada di subfolder data_preprocessing
    file_path = "data_preprocessing/loan_preprocessing.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: File tidak ditemukan di {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # Sesuaikan nama kolom target jika berbeda
    target_col = 'Loan_Approved'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Starting training...")
    with mlflow.start_run():
        # Setup Model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Log Metrics & Model ke DagsHub
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model trained with Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()