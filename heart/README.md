# ❤️ Análise de Dados — Heart Disease Dataset

Este notebook apresenta uma análise exploratória e preditiva sobre o conjunto de dados **Heart Disease** (Doença Cardíaca), aplicando técnicas de **Ciência de Dados** e **Machine Learning** para prever a presença de doença cardíaca com base em variáveis clínicas.

## 📊 Objetivo

Explorar, visualizar e modelar o conjunto de dados **Heart Disease**, com o objetivo de identificar padrões e construir modelos capazes de prever o diagnóstico da doença em pacientes.

## 🧰 Tecnologias Utilizadas

- **Python 3+**
- **Jupyter Notebook**
- **Pandas** — manipulação e análise de dados
- **NumPy** — operações matemáticas
- **Matplotlib** e **Seaborn** — visualização dos dados
- **Scikit-learn** — modelagem preditiva e avaliação de desempenho
    - `DecisionTreeClassifier`
    - `StandardScaler`
    - `train_test_split`, `cross_val_score`
    - `accuracy_score`, `classification_report`, `confusion_matrix`

## ⚙️ Etapas Realizadas

1. Carregamento e inspeção inicial dos dados (`heart.csv`).
2. Seleção das *features* (variáveis) `cp`, `thalach`, `slope` e da variável alvo `target`.
3. Avaliação inicial do modelo `DecisionTreeClassifier` usando validação cruzada (`cross_val_score`) para medir a performance média.
4. Normalização dos dados com `StandardScaler`.
5. Separação dos dados em conjuntos de treino e teste.
6. Treinamento do modelo final de Árvore de Decisão.
7. Avaliação detalhada do modelo com acurácia, *classification report* e uma matriz de confusão visualizada com o Seaborn.

## 🚀 Resultados e Conclusões

O modelo de **Árvore de Decisão** treinado demonstrou alta eficácia, alcançando uma **acurácia de aproximadamente 91,2%** no conjunto de teste.

As métricas de precisão (*precision*) e *recall* para ambas as classes (pacientes com e sem doença) foram robustas, indicando que o modelo é capaz de identificar corretamente a maioria dos casos. A análise focada nas *features* `cp` (tipo de dor no peito), `thalach` (frequência cardíaca máxima) e `slope` (inclinação do segmento ST) provou ser altamente preditiva para o diagnóstico de doenças cardíacas.
