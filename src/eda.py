# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
config.set_plot_style()

from sklearn import model_selection
from scipy.stats import chi2_contingency
from IPython.display import display
import statsmodels.api as sm

# %%
df_processed = pd.read_csv('../data/processed/processed_cardio_data.csv')
# %% 
target = 'flag_doenca_cardiaca'

# Identificar colunas categóricas e numéricas, removendo 'paciente_id'
cat_features = [col for col in df_processed.select_dtypes(include=['object']).columns if col != 'paciente_id'] + \
               [col for col in df_processed.select_dtypes(include=['int64', 'float64']).columns if col.startswith('flag_') and col not in ['paciente_id', target]]
num_features = [col for col in df_processed.select_dtypes(include=['int64', 'float64']).columns if not col.startswith('flag_') and col not in ['paciente_id', target]]

features = list(set(cat_features + num_features))

print("Variáveis categóricas: ", cat_features)
print("Variáveis numéricas: ", num_features)
print("Variável alvo: ", target)


# %%
df_processed[cat_features + [target]].describe()

# %%
df_processed[num_features].describe().T

# %%
# Análise de Correlação - Coeficiente de Pearson

corr_table = df_processed[num_features + [target]].corr()

mask = np.triu(np.ones_like(corr_table, dtype=np.bool_))

heatmap_corr_casas = sns.heatmap(corr_table, mask=mask, vmin=-1, vmax=1, cmap='BrBG', annot=True, annot_kws={"fontsize":8, "color":"k"})
heatmap_corr_casas.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=16)
plt.show()

# %%
# Distribuição das variáveis numéricas em relação à variável alvo
for var in num_features:
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot histogram on the first subplot
    sns.histplot(
        data=df_processed, 
        x=var, 
        hue=target, 
        kde=True, 
        bins=30, 
        palette={0: "gray", 1: "lightsalmon"},  
        hue_order=[0, 1],
        ax=axes[0],
        legend=False
    )
    axes[0].set_title(f'Distribuição de {var} em relação à {target}', fontsize=14)
    axes[0].set_xlabel(var)
    axes[0].set_ylabel('Frequência')

    # Plot boxplot on the second subplot
    sns.boxplot(
        data=df_processed, 
        y=df_processed[target].astype('category'), 
        x=var, 
        palette={'0': 'gray', '1': 'lightsalmon'},
        ax=axes[1]
    )
    axes[1].set_title(f'Análise de {var} em relação à target', fontsize=14)
    axes[1].set_xlabel(var)
    axes[1].set_ylabel('Target')
    axes[1].legend(loc='upper right', labels=['Ausência de Doença Cardíaca', 'Presença de Doença Cardíaca'])

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
# %%
# Point plot e distribuição de frequências para variáveis categóricas
for var in cat_features:
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.pointplot(data=df_processed, 
                  y=target,
                  x=var, 
                  ax=ax, 
                  capsize=.1,
                  color='lightcoral')
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    sns.countplot(data=df_processed, 
                  x=var, 
                  hue=var, 
                  palette={"gray"}, 
                  legend=False,
                  alpha=0.1, 
                  ax=ax2)
    ax2.set_ylabel('Frequência')
    ax2.tick_params(axis='y')

# %%
# Análise de Associação - Qui Quadrado de Associação
combinacoes_cat_target = [(col, target) for col in cat_features]

resultados_chi = []
tabelas_contigencia = []

for comb in combinacoes_cat_target:
    cross_tab = pd.crosstab(df_processed[comb[0]], df_processed[comb[1]], margins=True, margins_name="total", normalize=True)
    chi2, p, dof, expected = chi2_contingency(cross_tab.iloc[:-1, :-1])

    resultados_chi.append([
        comb,
        p,
        p <= 0.05
    ])

    tabelas_contigencia.append(cross_tab)

df_resultados_teste_chi2 = pd.DataFrame(resultados_chi, columns=['Variáveis', 'p_valor', 'Significativo'])
df_resultados_teste_chi2

# %%
# Heatmap de Resíduos Padronizados Ajustados
n = len(tabelas_contigencia[1:])
cols = 3
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for i, tabela in enumerate(tabelas_contigencia[1:]):
    tab_cont = sm.stats.Table(tabela)
    resids = tab_cont.standardized_resids

    sns.heatmap(
        data=resids,
        vmin=np.min(resids) - 0.1,
        vmax=np.max(resids) + 0.1,
        annot=resids.round(2),
        fmt=".2f",
        cmap=sns.color_palette(["white", "skyblue"], as_cmap=True),
        linewidths=.5,
        ax=axes[i]
    )

    axes[i].set_title(f'Resíduos Padronizados')

for j in range(n, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %%
for tabela_cont in tabelas_contigencia:
    styled_crosstab = tabela_cont.iloc[:-1, :-1].style.background_gradient(cmap='YlOrBr', axis=0)
    display(styled_crosstab)

# %%
# Retirada de observações com altura a baixo de 145cm

df_processed = df_processed[df_processed['vlr_altura'] >= 145]
df_processed.to_csv('../data/processed/processed_cardio_data.csv', index=False)