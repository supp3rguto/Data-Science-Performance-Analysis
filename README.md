# Projeto de Portfólio: Pipeline de Análise de Performance de Modelos de ML

Este repositório demonstra um pipeline completo de Data Science para avaliar e comparar a performance do algoritmo de **Árvore de Decisão (Decision Tree)** em oito conjuntos de dados distintos e heterogêneos.

O projeto vai além da simples aplicação de um modelo, focando em **engenharia de dados robusta**, **pré-processamento avançado** (incluindo NLP) e **análise crítica de resultados** — habilidades fundamentais em qualquer projeto de Machine Learning e Data Science.

## ✨ Foco em Habilidades

Este projeto não é apenas sobre Árvores de Decisão; é uma demonstração de competências essenciais para o mercado de Data Science:

1. **Fundamentos de NLP (Habilidade para LLMs):**
   O pré-processamento de texto nas bases `Spam` e `Wine` utilizando **TfidfVectorizer** demonstra a capacidade de converter texto não estruturado em features numéricas. Esta é a habilidade fundacional para tarefas modernas de NLP e essencial para o pipeline de qualquer modelo de linguagem (LLM).

2. **Engenharia de Dados Robusta:**
   A análise foi inicialmente tentada no Weka, mas a ferramenta falhou em lidar com os desafios do mundo real. Foi desenvolvido um **pipeline programático em Python** para superar:

   * **Erros de Memória (`OutOfMemoryError`):** Ao lidar com a base `Covertype` (+580k linhas), foi implementada **amostragem** para viabilizar a análise.
   * **Erros de Parsing:** O Python/Pandas foi usado para ler e tratar CSVs mal formatados (`Titanic`, `Spam`, `Wine`) que o Weka não conseguiu processar.

3. **Pré-processamento de Dados Mistos:**
   A base `Titanic` exigiu um tratamento complexo, incluindo:

   * **Imputação de Dados Faltantes:** Uso de média (`Age`, `Fare`) e moda (`Embarked`).
   * **Encoding Categórico:** Conversão de colunas como `Sex` e `Embarked` para formato numérico (`LabelEncoder`).

4. **Avaliação Rigorosa de Modelos:**
   A performance não foi medida com uma simples divisão de treino/teste. Foi utilizada **Validação Cruzada Estratificada de 10 Partições** (`StratifiedKFold`) para garantir uma estimativa realista e confiável da performance do modelo em dados nunca vistos.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Bibliotecas Principais:**

  * **Scikit-learn:** Para o modelo `DecisionTreeClassifier`, `StratifiedKFold` (validação), `TfidfVectorizer` (NLP) e `LabelEncoder` (encoding).
  * **Pandas:** Para leitura, limpeza, manipulação, imputação e amostragem de dados.
  * **Jupyter Notebook:** Ambiente de desenvolvimento e análise interativa.

## 📊 Resumo dos Resultados e Desafios

A performance do modelo variou significativamente entre as bases, provando que o sucesso de um modelo depende da adequação dos dados e do pré-processamento.

| Base de Dados | Tipo de Desafio                    | Acurácia Média |
| :------------ | :--------------------------------- | :------------: |
| **Heart**     | Numérico, 2 classes                |     99.51%     |
| **Spam**      | NLP (Texto), 2 classes             |     95.91%     |
| **Iris**      | Numérico, 3 classes                |     94.00%     |
| **Breast**    | Numérico, 2 classes                |     91.22%     |
| **Titanic**   | Misto (Num/Cat), 2 classes         |     77.29%     |
| **Covertype** | Big Data (Amostra), 7 classes      |     75.54%     |
| **Diabetes**  | Numérico (Sobreposição), 2 classes |     71.36%     |
| **Wine**      | NLP (Texto), 20+ classes           |     60.18%     |

## 📂 Estrutura do Repositório

Este repositório contém oito Jupyter Notebooks, cada um dedicado à análise de uma das bases de dados listadas acima.
Cada notebook possui seu próprio `README.md` detalhando o processo específico de pré-processamento e os resultados da análise.

```
├── jpntbkpython_breast.ipynb
├── jpntbkpython_covertype.ipynb
├── jpntbkpython_diabets.ipynb
├── jpntbkpython_heart.ipynb
├── jpntbkpython_iris.ipynb
├── jpntbkpython_spam.ipynb
├── jpntbkpython_titanic.ipynb
└── jpntbkpython_wine.ipynb
```
## 👨‍💻 Autor

**Augusto Ortigoso Barbosa**

  * **GitHub:** [github.com/supp3rguto](https://github.com/supp3rguto)
  * **LinkedIn:** [linkedin.com/in/augusto-barbosa-769602194](https://www.linkedin.com/in/augusto-barbosa-769602194)
