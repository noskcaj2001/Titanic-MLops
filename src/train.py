import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(train_path):
    """Treina um modelo e rastreia o experimento com MLflow."""
    # Carregar dados
    train = pd.read_csv(train_path)
    
    # Separar features e target
    X = train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = train['Survived']
    
    # Dividir em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configurar MLflow
    mlflow.set_experiment("Titanic-Survival")
    
    # Habilitar autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        # Treinar modelo
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Fazer previsões
        y_pred = model.predict(X_val)
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        # Logar métricas manualmente (opcional, pois o autolog já faz isso)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Salvar o ID do run em um arquivo
        with open('models/last_run_id.txt', 'w') as f:
            f.write(mlflow.active_run().info.run_id)
        
        print(f"Modelo treinado. Métricas: {metrics}")
        return model

if __name__ == "__main__":
    train_model('data/processed/train_featured.csv')