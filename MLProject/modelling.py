import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- KONFIGURASI ---
DAGSHUB_USERNAME = "tickelboy" 
DAGSHUB_REPO_NAME = "Eksperimen_SML_StanlyLopez"

# --- LOGIC OTENTIKASI ---
token = os.getenv("DAGSHUB_TOKEN")
if token:
    print(f"CI/CD Mode: Login manual ke {DAGSHUB_REPO_NAME}")
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(f"https://dagshub.com/Tikleboy/Eksperimen_SML_StanlyLopez.mlflow")
else:
    print("Local Mode: Menggunakan dagshub.init()")
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

mlflow.set_experiment("CI_Pipeline_Experiment")

def train_model():
    print("Loading data...")
    # Path disesuaikan karena script dijalankan di dalam folder MLProject
    file_path = "data_preprocessing/loan_preprocessing.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} tidak ditemukan.")
        return

    df = pd.read_csv(file_path)
    X = df.drop(columns=['Loan_Approved'])
    y = df['Loan_Approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Starting training...")
    
    # Tangkap object 'run' agar kita bisa ambil ID-nya
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        
        # 1. SIMPAN MODEL DENGAN NAMA 'model_final' (Sesuai request YAML Anda)
        mlflow.sklearn.log_model(model, "model_final")
        print(f"Model saved as 'model_final'. Accuracy: {acc:.4f}")
        
        # 2. SIMPAN RUN ID KE FILE TEXT (PENTING UNTUK YAML)
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Run ID {run_id} saved to run_id.txt")

if __name__ == "__main__":
    train_model()
