# Titanic Survival Prediction

## Descrição do Projeto
Este projeto tem como objetivo prever a sobrevivência de passageiros no naufrágio do Titanic utilizando técnicas de machine learning. Além de ser um desafio clássico de ciência de dados, o projeto também busca aplicar boas práticas de engenharia de software para colocação de modelos em produção de forma escalável e sustentável.

## Objetivos
### Previsão de Sobrevivência
Desenvolver um modelo de machine learning para prever se um passageiro do Titanic sobreviveu ou não, com base em características como idade, sexo, classe, entre outras.

### Engenharia de Software
- Estruturar o projeto de machine learning de forma modular e escalável.
- Utilizar boas práticas de engenharia de software para facilitar a manutenção, colaboração e implantação do modelo em produção.

### MLOps
- Integrar o MLflow para rastrear experimentos, versionar modelos e monitorar métricas.
- Preparar o pipeline para ser escalável e replicável em ambientes de produção.

## Estrutura do Projeto
```
titanic-kaggle/
├── data/
│   ├── raw/            # Arquivos originais do Kaggle
│   └── processed/      # Dados processados
├── src/
│   ├── preprocess.py   # Pré-processamento dos dados
│   ├── feature_engineering.py  # Engenharia de features
│   ├── train.py        # Treinamento de modelos
│   └── predict.py      # Geração de previsões
├── mlruns/             # Diretório do MLflow
├── models/             # Modelos salvos
├── submissions/        # Arquivos de submissão para o Kaggle
└── README.md           # Documentação do projeto
```

## Engenharia de Software Aplicada
### 1. Modularização
O projeto é dividido em módulos independentes:
- **Pré-processamento:** Limpeza e tratamento de dados.
- **Engenharia de Features:** Criação de novas variáveis e transformações.
- **Treinamento:** Treinamento e avaliação de modelos.
- **Previsão:** Geração de previsões para novos dados.

Isso facilita a manutenção, testes e reutilização do código.

### 2. Rastreamento de Experimentos com MLflow
O MLflow é utilizado para:
- Registrar parâmetros, métricas e artefatos de cada experimento.
- Versionar modelos treinados.
- Facilitar a comparação entre diferentes abordagens.

### 3. Pipeline Reproduzível
O pipeline é executado em etapas claras e independentes:
1. Pré-processamento.
2. Engenharia de features.
3. Treinamento do modelo.
4. Geração de previsões.

Cada etapa pode ser executada separadamente, facilitando a depuração e a colaboração.

### 4. Escalabilidade
O uso de ferramentas como MLflow e a modularização do código permitem que o projeto seja facilmente escalado para:
- Maiores volumes de dados.
- Novos modelos e técnicas de machine learning.
- Implantação em ambientes de produção.

### 5. Versionamento de Modelos
O MLflow permite salvar e versionar modelos treinados, garantindo que seja possível reproduzir resultados e reutilizar modelos em diferentes cenários.

## Como Executar o Projeto
### Pré-requisitos
- Python 3.8+
- Bibliotecas listadas em `requirements.txt`.

Instale as dependências:
```bash
pip install -r requirements.txt
```

### Passo a Passo
#### main:
```bash
python main.py
```

#### Visualizar Experimentos no MLflow:

```bash
mlflow ui
```
Acesse `http://localhost:5000` para ver os experimentos e modelos registrados.

## Dados
Os dados utilizados são os arquivos `train.csv` e `test.csv` do desafio Titanic: Machine Learning from Disaster no Kaggle. Eles devem ser colocados na pasta `data/raw/`.

## Resultados
### Arquivo de Submissão:
As previsões são salvas em `submissions/submission.csv` e podem ser enviadas ao Kaggle.

### Modelos Registrados:
Todos os modelos treinados são registrados no MLflow, permitindo comparação e reutilização.

## Próximos Passos
### Automatização do Pipeline:
Utilizar ferramentas como Airflow ou Prefect para orquestrar o pipeline de forma automatizada.

### Implantação em Produção:
- Integrar o modelo treinado em uma API REST usando frameworks como FastAPI ou Flask.
- Utilizar contêineres Docker para facilitar a implantação em diferentes ambientes.

### Monitoramento de Modelos:
Implementar monitoramento contínuo do desempenho do modelo em produção.

## Autor
[Jackson Costa]

## Licença
Este projeto está licenciado sob a Licença MIT.

## Referências
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [MLflow Documentation](https://mlflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

Este projeto é um exemplo prático de como aplicar boas práticas de engenharia de software e MLOps para criar soluções de machine learning escaláveis e prontas para produção. 🚀# Titanic-MLops
