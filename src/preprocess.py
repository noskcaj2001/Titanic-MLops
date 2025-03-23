import pandas as pd

def preprocess_data(train_path, test_path):
    """Carrega e pré-processa os dados."""
    # Carregar dados
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Preencher valores faltantes
    train['Age'].fillna(train['Age'].median(), inplace=True)
    test['Age'].fillna(test['Age'].median(), inplace=True)
    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)
    
    # Salvar dados processados
    train.to_csv('data/processed/train_processed.csv', index=False)
    test.to_csv('data/processed/test_processed.csv', index=False)
    
    print("Dados pré-processados salvos em data/processed/")
    return train, test

if __name__ == "__main__":
    preprocess_data('data/raw/train.csv', 'data/raw/test.csv')