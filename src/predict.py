import pandas as pd
import mlflow
import mlflow.sklearn

def predict(test_path, output_path):
    """Gera previsões para o conjunto de teste."""
    # Carregar dados
    test = pd.read_csv(test_path)
    
    # Carregar o ID do run salvo
    try:
        with open('models/last_run_id.txt', 'r') as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("Erro: Nenhum modelo treinado encontrado. Execute o script de treinamento primeiro.")
        return
    
    # Carregar modelo
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Fazer previsões
    passenger_ids = test['PassengerId']
    X_test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    predictions = model.predict(X_test)
    
    # Salvar previsões
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    submission.to_csv(output_path, index=False)
    
    print(f"Previsões salvas em {output_path}")

if __name__ == "__main__":
    predict('data/processed/test_featured.csv', 'submissions/submission.csv')