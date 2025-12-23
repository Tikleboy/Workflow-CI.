import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- KONFIGURASI USER ---
DAGSHUB_USERNAME = "Tickelboy"
DAGSHUB_REPO_NAME = "Eksperimen_SML_StanlyLopez" 

# --- LOGIC OTENTIKASI ---
token = os.getenv("DAGSHUB_TOKEN")

if token:
    print(f"Terdeteksi CI/CD Environment. Menggunakan Login Token Manual ke {DAGSHUB_REPO_NAME}")
    # Cara Manual: Lebih stabil untuk CI/CD karena tidak memicu browser login
    uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(uri)
    
    # Set kredensial secara eksplisit ke Environment Variables
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
else:
    print("Terdeteksi Lokal Environment. Menggunakan dagshub.init()")
    # Cara Lokal: Praktis untuk di laptop
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

mlflow.set_experiment("CI_Pipeline_Experiment")

def train_model():
    print("Loading data...")
    # Path file (Pastikan file ada di folder data_preprocessing dalam MLProject)
    file_path = "data_preprocessing/loan_preprocessing.csv"
    
    # Sedikit error handling agar path aman
    if not os.path.exists(file_path):
        print(f"Error: File tidak ditemukan di {file_path}")
        print(f"Isi folder saat ini: {os.listdir('.')}")
        return

    df = pd.read_csv(file_path)
    
    target_col = 'Loan_Approved'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Starting training...")
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model trained successfully! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()
