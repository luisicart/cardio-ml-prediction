# cardio-ml-prediction

Este projeto tem como objetivo desenvolver e avaliar modelos preditivos para a ocorrência de doenças cardíacas utilizando dados abertos, com ênfase na identificação das variáveis mais influentes na sua previsão. Ele faz parte do Trabalho de Conclusão de Curso do MBA em Ciência de Dados da USP/Esalq.

Este tema surgiu em minha vida por diferentes motivos, onde culminaram no período de conclusão do curso em Data Science. A ideia inicial está relacionada a desafios enfrentados dentro de uma empresa de Seguros de Vida, onde nossa maior dificuldade era entender o risco atrelado a um potencial cliente de acordo seu histórico médico. O entendimento dessas circunstâncias são essencial para a precificação e aceite do risco. 

O outro motivador é de cunho pessoal, já que possuo em minha família uma pessoa que possui doença cardíaca e sua previsão seria um fator crucial para o tratamento. 

Além do objetivo central do trabalho ser a escolha dos melhores algoritmos para previsão e entender o impacto de cada variável, é de suma importância relembrar que este projeto só existe para que possamos auxiliar na previsão e tomada de decisão para cada tipo de grupo de risco, seja por intervenção ou a necessidade de novos exames.

Os dados utilizados são uma consolidação de duas bases:
* ["UCI Machine Learning Repository - Heart Disease Dataset"](https://archive.ics.uci.edu/dataset/45/heart+disease)
* ["Kaggle - Heart Disease Dataset by YasserH"](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset)

---

## Estrutura do Projeto

A estrutura do projeto visa a modularidade e reprodutibilidade por qualquer pessoa que queira realizar suas previsões sobre esta base de dados ou outras. 
Portanto, está dividido em pastas que possuem diferentes propósitos que são autoexplicados em seus nomes.

A pasta `src` é onde estão as principais etapas de análise exploratória, transformação dos dados, treinamento, avaliação dos modelos e interpretabilidade.


---

## Tecnologias Utilizadas

- Python 3.10+
- Pandas, NumPy
- Seaborn & Matplotlib
- Scikit-learn
- XGBoost
- CatBoost
- Optuna 
- SHAP

---

## Modelos Implementados

- Regressão Logística
- Decision Tree
- Random Forest
- XGBoost
- CatBoost
- Adaboost

---

## Avaliação dos Modelos

As seguintes métricas são utilizadas para avaliar a performance dos modelos:

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**
- **Área sob a Curva ROC (AUC-ROC)**
- **Matriz de confusão**

As métricas utilizadas foram selecionadas por fornecerem uma visão abrangente do desempenho dos modelos, especialmente em contextos com possível desbalanceamento de classes, como é comum em diagnósticos médicos.

---

## Interpretabilidade com SHAP

A análise de explicabilidade dos modelos é feita com **SHAP (Shapley Additive Explanations)**, permitindo:
- Visualizar o impacto de cada variável na predição
- Identificar os atributos mais importantes
- Aumentar a transparência e a confiança no modelo

O SHAP (Shapley Additive Explanations) foi escolhido por ser um dos métodos mais robustos e interpretáveis para explicar predições de modelos de machine learning. Ele se baseia na teoria dos jogos para atribuir a cada variável a sua real contribuição para a predição, oferecendo explicações consistentes e localmente precisas, mesmo em modelos complexos como árvores de decisão ou ensembles. Isso permite compreender o "porquê" por trás das decisões do modelo, aumentando a confiança e a transparência (aspectos fundamentais em aplicações na área da saúde).

---

## Como Executar o Projeto

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

## Dataset

Nomes e conteúdo de variáveis foram substituídos para o português, para que seja mais clara sua interpretabilidade para pessoas leigas no assunto. A versão original pode ser encontrada nos links disponibilizados e o script de transformação na pasta `src`.

**Variáveis:**
* paciente_id: Identificador único de cada paciente.
* nr_dias_idade: Idade do paciente em dias.
* nr_anos_idade: Idade do paciente em anos (derivada da variável "age").
* cat_genero: Gênero do paciente - Variável categórica (1: Feminino, 2: Masculino).
* vlr_altura: Altura do paciente em centímetros.
* vlr_peso: Peso do paciente em kilogramas.
* vlr_pressao_sistolica: Pressão arterial Sistólica.
    * A pressão arterial sistólica é o valor mais alto registrado durante a medição da pressão arterial e representa a pressão do sangue nas artérias quando o coração se contrai (sístole). Os valores normais para a pressão sistólica em adultos geralmente variam entre 120 e 129 mmHg, sendo que abaixo de 120 mmHg é considerado ótimo. 
* vlr_pressão_diastolica: Pressão arterial Diastólica.
    * Refere-se a pressão do sangue nas artérias quando o coração está em repouso, entre os batimentos, e é o valor inferior da leitura da pressão arterial. É medida em milímetros de mercúrio (mmHg) e é considerada normal entre 60 e 80 mmHg em adultos
* cat_colesterol: Níveis de Colesterol. Variável categórica (1: Normal, 2: Acima do Normal, 3: Bem Acima do Normal).
* cat_glicose: Níveis de Glicose no sanguie. Variável categórica (1: Normal, 2: Acima do Normal, 3: Bem Acima do Normal).
* flag_fumante: Status de Fumante. Variável binária (0: Não fumante, 1: Fumante).
* flag_consumo_alcool: Ingestão de Álcool. Variável binária (0: Não consome, 1: Consome).
* flag_atividade_fisica: Atividade física. Variável binária (0: Não pratica, 1: Pratica).
* vlr_imc: Índice IMC, calculado através do peso e altura. Calculado da seguinte maneira: 
    * IMC = peso (kg) \ altura (m)^2
* cat_pressao_arterial: Categorização da pressão arterial, derivado das variáveis ap_hi e ap_lo. Variável categórica ("Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2", and "Hypertensive Crisis").
* flag_doenca_cardiaca: Presença ou ausência de doença cardíaca. **Variável Target**. Binária (0: Ausência, 1: Presença).