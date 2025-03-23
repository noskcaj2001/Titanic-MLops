import pandas as pd

def feature_engineering(train_path, test_path):
    """Realiza engenharia de features."""
    # Carregar dados pré-processados
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Criar FamilySize
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    
    # Criar IsAlone
    train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
    test['IsAlone'] = (test['FamilySize'] == 1).astype(int)
    
    # Codificar variáveis categóricas
    train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
    test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Salvar dados com features
    train.to_csv('data/processed/train_featured.csv', index=False)
    test.to_csv('data/processed/test_featured.csv', index=False)
    
    print("Engenharia de features concluída. Dados salvos em data/processed/")
    return train, test

if __name__ == "__main__":
    feature_engineering('data/processed/train_processed.csv', 'data/processed/test_processed.csv')