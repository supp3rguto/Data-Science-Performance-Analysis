# 🍷 Análise de Dados — Wine Review Dataset

Este notebook apresenta uma análise preditiva sobre o conjunto de dados **Wine Reviews**, aplicando **Machine Learning** para classificar o país de origem de um vinho com base em suas características, como preço, pontuação e variedade.

## 📊 Objetivo

Explorar, tratar e modelar o conjunto de dados de vinhos, com o objetivo de construir um modelo de classificação capaz de prever o **país (`country`)** de origem do vinho. A análise foi focada nos **5 países com mais amostras** no dataset.

## 🧰 Tecnologias Utilizadas

- **Python 3+**
- **Jupyter Notebook**
- **Pandas** — manipulação e análise de dados
- **NumPy** — operações matemáticas
- **Matplotlib** e **Seaborn** — visualização dos dados
- **Scikit-learn** — pré-processamento, modelagem e avaliação
    - `DecisionTreeClassifier`
    - `LabelEncoder` (para features categóricas e alvo)
    - `StandardScaler` (para normalização)
    - `train_test_split`, `cross_val_score`
    - `accuracy_score`, `classification_report`, `confusion_matrix`

## ⚙️ Etapas Realizadas

1.  Carregamento do dataset (`wine.csv`).
2.  Filtragem do dataset para incluir apenas os **Top 5** países com mais registros.
3.  Seleção das *features* (`points`, `price`, `province`, `region_1`, `variety`, `winery`) e do alvo (`country`).
4.  **Tratamento de Dados Ausentes:** Preenchimento de valores nulos usando a **mediana** para colunas numéricas (`price`) e a **moda** para colunas categóricas.
5.  **Pré-processamento (Encoding):** Aplicação do `LabelEncoder` em todas as *features* categóricas e na variável alvo.
6.  **Avaliação (Cross-Validation):** Teste inicial de um `DecisionTreeClassifier` (critério 'entropy') com validação cruzada (10 folds).
7.  **Pré-processamento (Scaling):** Normalização das *features* (X) com `StandardScaler`.
8.  Divisão dos dados normalizados em conjuntos de treino (70%) e teste (30%).
9.  Treinamento do modelo final de Árvore de Decisão (`random_state=42`).
10. Avaliação detalhada do modelo no conjunto de teste (Acurácia, Classification Report e Matriz de Confusão).

## 🚀 Resultados e Conclusões

O modelo de **Árvore de Decisão** demonstrou uma performance **extremamente alta**, alcançando uma **acurácia de 99,98%** tanto na validação cruzada quanto no conjunto de teste final.

A matriz de confusão e o *classification report* confirmam que o modelo é praticamente perfeito, com quase nenhuma classificação incorreta (apenas 4 erros em 32.542 amostras de teste).

Este desempenho excepcional sugere que *features* como `winery` (vinícola), `region_1` (região) e `province` (província) são preditores muito fortes (quase determinísticos) para o `country`, pois geralmente existe uma relação direta e exclusiva entre essas localizações e o país de origem.
