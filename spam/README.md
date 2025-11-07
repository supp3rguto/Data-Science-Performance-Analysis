# 🚫 Análise de Dados — Spam Detection Dataset

Este notebook apresenta uma análise preditiva e técnicas de **Processamento de Linguagem Natural (NLP)** sobre um conjunto de dados de **Spam**, aplicando **Machine Learning** para classificar textos como "spam" ou "ham" (não-spam).

## 📊 Objetivo

Explorar e modelar o conjunto de dados **Spam**, com o objetivo de construir um modelo de classificação capaz de prever com precisão se uma nova mensagem de texto é ou não spam, com base em seu conteúdo.

## 🧰 Tecnologias Utilizadas

- **Python 3+**
- **Jupyter Notebook**
- **Pandas** — manipulação e análise de dados
- **NumPy** — operações matemáticas
- **Matplotlib** e **Seaborn** — visualização dos dados
- **Scikit-learn** — modelagem preditiva e processamento de texto
    - `TfidfVectorizer` (para vetorização de texto)
    - `LabelEncoder` (para a variável alvo)
    - `DecisionTreeClassifier` (modelo de classificação)
    - `train_test_split`, `cross_val_score`
    - `accuracy_score`, `classification_report`, `confusion_matrix`

## ⚙️ Etapas Realizadas

1.  Carregamento e inspeção dos dados (`spam.csv`).
2.  Processamento de Texto (NLP): Transformação da coluna `text` em uma matriz numérica de *features* usando `TfidfVectorizer`, limitando a 5000 *features* e removendo *stop words*.
3.  Codificação da variável alvo (`text_type`) de categórica para numérica ('ham' -> 0, 'spam' -> 1) com `LabelEncoder`.
4.  Avaliação inicial do modelo `DecisionTreeClassifier` usando validação cruzada (`cross_val_score`) para estimar a performance média.
5.  Separação dos dados vetorizados em conjuntos de treino e teste.
6.  Treinamento do modelo final de Árvore de Decisão (`random_state=42`).
7.  Avaliação detalhada do desempenho do modelo no conjunto de teste, incluindo acurácia, *classification report* (precision, recall, f1-score) e uma matriz de confusão visualizada.

## 🚀 Resultados e Conclusões

O modelo de **Árvore de Decisão**, combinado com a vetorização **TF-IDF**, mostrou-se eficaz na detecção de spam, alcançando uma **acurácia de aproximadamente 90,5%** no conjunto de teste.

A validação cruzada inicial já apontava uma acurácia média de ~85,6%. O *classification report* final indica um bom equilíbrio entre *precision* e *recall* para ambas as classes, com o modelo sendo ligeiramente melhor em identificar "ham" (classe 0) do que "spam" (classe 1). A matriz de confusão confirma que o modelo classifica corretamente a grande maioria das mensagens.
