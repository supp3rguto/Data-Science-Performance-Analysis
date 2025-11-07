# 🚢 Análise de Dados — Titanic Dataset

Este notebook apresenta uma análise exploratória e a construção de um modelo preditivo sobre o clássico conjunto de dados **Titanic**, utilizando **Machine Learning** para tentar prever a sobrevivência dos passageiros.

## 📊 Objetivo

Explorar, tratar e modelar o conjunto de dados **Titanic**, com o objetivo de construir um classificador capaz de prever se um passageiro **sobreviveu** (`Survived`) ou não, com base em suas características demográficas e de viagem.

## 🧰 Tecnologias Utilizadas

- **Python 3+**
- **Jupyter Notebook**
- **Pandas** — manipulação e análise de dados
- **NumPy** — operações matemáticas
- **Matplotlib** e **Seaborn** — visualização dos dados
- **Scikit-learn** — pré-processamento, modelagem e avaliação
    - `DecisionTreeClassifier`
    - `LabelEncoder`, `StandardScaler`
    - `train_test_split`, `cross_val_score`
    - `accuracy_score`, `classification_report`, `confusion_matrix`

## ⚙️ Etapas Realizadas

1.  Carregamento do dataset (`titanic.csv`).
2.  Seleção de *features* para o modelo (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`) e da variável alvo (`Survived`).
3.  **Tratamento de Dados Ausentes:**
    * `Age` e `Fare` preenchidos com a **mediana**.
    * `Embarked` preenchido com a **moda**.
4.  **Pré-processamento:**
    * `LabelEncoder` aplicado às colunas categóricas `Sex` e `Embarked`.
5.  **Modelagem e Avaliação (Etapa 1 - Cross-Validation):**
    * As *features* (X) foram definidas **removendo `Survived` e `Sex`**.
    * Um `DecisionTreeClassifier` (critério 'entropy') foi avaliado com `cross_val_score` (10 folds), obtendo uma acurácia média de ~62,5%.
6.  **Modelagem e Avaliação (Etapa 2 - Teste Final):**
    * Os dados (X) foram normalizados com `StandardScaler`.
    * Os dados foram divididos em conjuntos de treino (70%) e teste (30%).
    * Um modelo final `DecisionTreeClassifier` (`random_state=42`) foi treinado.
    * O modelo foi avaliado no conjunto de teste, incluindo acurácia, *classification report* e matriz de confusão.

## 🚀 Resultados e Conclusões

O modelo final de Árvore de Decisão apresentou um **desempenho limitado**, alcançando uma **acurácia de aproximadamente 59,5%** no conjunto de teste.

O *classification report* detalhado e a matriz de confusão revelam a principal dificuldade do modelo: ele possui um *recall* (0.29) e *precision* (0.35) muito baixos para a classe "Sobreviveu" (1). Isso indica que, embora tenha uma performance razoável para prever quem *não* sobreviveu (classe 0), o modelo falha em identificar corretamente os passageiros que sobreviveram.

O baixo desempenho geral, inferior ao de *benchmarks* comuns para este dataset, é provavelmente influenciado pela decisão de **remover a feature `Sex`** das *features* (X) de treinamento na etapa 5.
