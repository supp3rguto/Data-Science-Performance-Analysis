# 🌸 Análise de Dados — Iris Dataset

Este notebook apresenta uma análise exploratória e um modelo preditivo sobre o clássico conjunto de dados **Iris**, aplicando técnicas de **Ciência de Dados** e **Machine Learning** para classificar as espécies de flores Iris com base em suas características morfológicas.

## 📊 Objetivo

Explorar, visualizar e modelar o conjunto de dados **Iris**, com o objetivo de construir um modelo de classificação capaz de prever a espécie da flor (Iris-setosa, Iris-versicolor ou Iris-virginica).

## 🧰 Tecnologias Utilizadas

- **Python 3+**
- **Jupyter Notebook**
- **Pandas** — manipulação e análise de dados
- **NumPy** — operações matemáticas
- **Matplotlib** e **Seaborn** — visualização dos dados
- **Scikit-learn** — modelagem preditiva e avaliação
    - `DecisionTreeClassifier`
    - `LabelEncoder` (para a variável alvo)
    - `StandardScaler` (para normalização)
    - `train_test_split`, `cross_val_score`
    - `accuracy_score`, `classification_report`, `confusion_matrix`

## ⚙️ Etapas Realizadas

1.  Carregamento e inspeção inicial dos dados (`iris.csv`).
2.  Codificação da variável alvo categórica (`species`) para valores numéricos usando `LabelEncoder`.
3.  Separação das *features* (X) e da variável alvo (y).
4.  Avaliação inicial do modelo `DecisionTreeClassifier` (com critério 'entropy') usando validação cruzada (`cross_val_score`) para estimar a performance.
5.  Normalização das *features* (X) usando `StandardScaler`.
6.  Divisão dos dados normalizados em conjuntos de treino e teste.
7.  Treinamento do modelo final de Árvore de Decisão nos dados de treino.
8.  Avaliação detalhada do modelo no conjunto de teste com acurácia, *classification report* e uma matriz de confusão visualizada.

## 🚀 Resultados e Conclusões

O modelo de **Árvore de Decisão** treinado apresentou um desempenho excelente, alcançando uma **acurácia de aproximadamente 95,6%** no conjunto de teste.

A validação cruzada inicial já indicava uma performance robusta, com acurácia média de 95,3%. O *classification report* final e a matriz de confusão mostram que o modelo consegue prever a espécie "Iris-setosa" (classe 0) com perfeição e tem altíssima precisão na distinção entre as outras duas espécies, cometendo poucos erros.
