# cardio-ml-prediction

Este projeto tem como objetivo desenvolver e avaliar modelos preditivos para a ocorrência de doenças cardíacas utilizando dados abertos, com ênfase na identificação das variáveis mais influentes na sua previsão. Ele faz parte do Trabalho de Conclusão de Curso do MBA em Ciência de Dados da USP/Esalq.

Este tema surgiu em minha vida por diferentes motivos, onde culminaram no período de conclusão do curso em Data Science. A ideia inicial está relacionada a desafios enfrentados dentro de uma empresa de Seguros de Vida, onde nossa maior dificuldade era entender o risco atrelado a um potencial cliente de acordo seu histórico médico. O entendimento dessas circunstâncias são essencial para a precificação e aceite do risco. 

O outro motivador é de cunho pessoal, já que possuo em minha família uma pessoa que possui doença cardíaca e sua previsão seria um fator crucial para o tratamento. 

Além do objetivo central do trabalho ser a escolha dos melhores algoritmos para previsão e entender o impacto de cada variável, é de suma importância relembrar que este projeto só existe para que possamos auxiliar na previsão e tomada de decisão para cada tipo de grupo de risco, seja por intervenção ou a necessidade de novos exames.

Os dados utilizados são uma consolidação de duas bases:
* ["UCI Machine Learning Repository - Heart Disease Dataset"](https://archive.ics.uci.edu/dataset/45/heart+disease)
* ["Kaggle - Heart Disease Dataset by YasserH"](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset)

---

## 📁 Estrutura do Projeto

A estrutura do projeto visa a modularidade e reprodutibilidade por qualquer pessoa que queira realizar suas previsões sobre esta base de dados ou outras. 
Portanto, está dividido em pastas que possuem diferentes propósitos que são autoexplicados em seus nomes.

A pasta `src` é onde estão as principais etapas de análise exploratória, transformação dos dados, treinamento, avaliação dos modelos e interpretabilidade.


---

## ⚙️ Tecnologias Utilizadas

- Python 3.10+
- Pandas, NumPy
- Seaborn & Matplotlib
- Scikit-learn
- XGBoost
- CatBoost
- Optuna 
- SHAP

---

## 🧠 Modelos Implementados

- Regressão Logística
- Decision Tree
- Random Forest
- XGBoost
- CatBoost
- Adaboost

---

## 🧪 Avaliação dos Modelos

As seguintes métricas são utilizadas para avaliar a performance dos modelos:

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**
- **Área sob a Curva ROC (AUC-ROC)**
- **Matriz de confusão**

As métricas utilizadas foram selecionadas por fornecerem uma visão abrangente do desempenho dos modelos, especialmente em contextos com possível desbalanceamento de classes, como é comum em diagnósticos médicos.

---

## 🔍 Interpretabilidade com SHAP

A análise de explicabilidade dos modelos é feita com **SHAP (Shapley Additive Explanations)**, permitindo:
- Visualizar o impacto de cada variável na predição
- Identificar os atributos mais importantes
- Aumentar a transparência e a confiança no modelo

O SHAP (Shapley Additive Explanations) foi escolhido por ser um dos métodos mais robustos e interpretáveis para explicar predições de modelos de machine learning. Ele se baseia na teoria dos jogos para atribuir a cada variável a sua real contribuição para a predição, oferecendo explicações consistentes e localmente precisas, mesmo em modelos complexos como árvores de decisão ou ensembles. Isso permite compreender o "porquê" por trás das decisões do modelo, aumentando a confiança e a transparência (aspectos fundamentais em aplicações na área da saúde).

---

## 📈 Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/heart-disease-prediction.git <nome-da-pasta>
cd <nome-da-pasta>
```

### 2. Instale os pacotes e Dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o pipeline completo

```bash
python main.py
```
