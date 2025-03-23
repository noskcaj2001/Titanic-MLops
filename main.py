import os
from src.preprocess import preprocess_data
from src.feature_engineering import feature_engineering
from src.train import train_model
from src.predict import predict

def run_pipeline():
    """Executa o pipeline completo de machine learning."""
    print("=== INICIANDO PIPELINE DO TITANIC ===")

    # 1. Pré-processamento
    print("\n=== ETAPA 1: PRÉ-PROCESSAMENTO ===")
    train_processed, test_processed = preprocess_data('data/raw/train.csv', 'data/raw/test.csv')
    print("Pré-processamento concluído. Dados salvos em data/processed/.")

    # 2. Engenharia de Features
    print("\n=== ETAPA 2: ENGENHARIA DE FEATURES ===")
    train_featured, test_featured = feature_engineering('data/processed/train_processed.csv', 'data/processed/test_processed.csv')
    print("Engenharia de features concluída. Dados salvos em data/processed/.")

    # 3. Treinamento do Modelo
    print("\n=== ETAPA 3: TREINAMENTO DO MODELO ===")
    train_model('data/processed/train_featured.csv')
    print("Treinamento concluído. Modelo salvo no MLflow.")

    # 4. Geração de Previsões
    print("\n=== ETAPA 4: GERAÇÃO DE PREVISÕES ===")
    predict('data/processed/test_featured.csv', 'submissions/submission.csv')
    print("Previsões geradas e salvas em submissions/submission.csv.")

    print("\n=== PIPELINE CONCLUÍDO COM SUCESSO! ===")

if __name__ == "__main__":
    # Criar diretórios necessários
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)

    # Executar o pipeline
    run_pipeline()