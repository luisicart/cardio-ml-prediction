# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
config.set_plot_style()

from scipy.stats import chi2_contingency
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
df_processed[num_features].describe().T.round(1)

# %%
# Análise de Percentis - Variáveis Numéricas (Outliers)
percentis = np.arange(0.01, 1.01, 0.01) 

percentis_peso = df_processed['vlr_peso'].quantile(percentis)
percentis_altura = df_processed['vlr_altura'].quantile(percentis)

df_percentis = pd.DataFrame({
    'percentil': [f'{int(p*100)}%' for p in percentis],
    'vlr_peso': percentis_peso.values,
    'vlr_altura': percentis_altura.values
})

df_percentis
 # %%
# Remoção de Outliers 
df_processed = df_processed[(df_processed['vlr_altura'] > 148) & (df_processed['vlr_peso'] > 48)].copy()
df_processed.info()

# %%
df_processed[num_features].describe().T.round(1)

# %% 
# Análise de Frequência - Variáveis Categóricas
frequencia_categorica = []
for var in cat_features:
    
    tab = pd.crosstab(df_processed[var], df_processed['flag_doenca_cardiaca'])
    
    tab = tab.rename(columns={0: 'contage_sem_doenca', 1: 'contage_com_doenca'}).copy()
    tab['contage_sem_doenca'] = tab.get('contage_sem_doenca', 0)
    tab['contage_com_doenca'] = tab.get('contage_com_doenca', 0)
    
    tab['total'] = tab['contage_sem_doenca'] + tab['contage_com_doenca']
    tab['%_sem_doenca'] = 100 * tab['contage_sem_doenca'] / tab['total']
    tab['%_com_doenca'] = 100 * tab['contage_com_doenca'] / tab['total']
    
    tab.reset_index(inplace=True)
    tab['variavel'] = var
    tab.rename(columns={var: 'valor'}, inplace=True)
    
    frequencia_categorica.append(tab[['variavel', 'valor', 'contage_sem_doenca', '%_sem_doenca', 'contage_com_doenca', '%_com_doenca']])

df_freq_categorica = pd.concat(frequencia_categorica, ignore_index=True)

df_freq_categorica.round(1)

# %%
# Análise de Associação - Teste Qui-Quadrado para variáveis categóricas
cat_table = pd.DataFrame(columns=['pvalue'])
cat_table_cont = pd.DataFrame()

for cat in cat_features:
    cont_table = df_processed.groupby(cat)['flag_doenca_cardiaca'].value_counts().unstack(fill_value=0)
    
    chi2, p, dof, ex = chi2_contingency(cont_table)
    p_val = round(p, 2)
    p_val = str(p_val) if p_val >= 0.01 else '<0.01'
    
    cat_table.loc[cat, 'pvalue'] = p_val
    
    cont_table.reset_index(inplace=True)
    cont_table.rename(columns={cat: 'category'}, inplace=True)
    cont_table.index = [cat] * cont_table.shape[0]
    
    cat_table_cont = pd.concat([cat_table_cont, cont_table], ignore_index=False)

cat_table = cat_table_cont.join(cat_table, how='left')
cat_table['total'] = cat_table[0] + cat_table[1]
cat_table = cat_table.reindex(columns=['category', 0, 1, 'total', 'pvalue'])

cat_table
# %%
# Análise de Correlação - Coeficiente de Pearson

corr_table = df_processed[num_features + [target]].corr()

# Formatar os números decimais com vírgula
fmt = lambda x: str(x).replace('.', ',') if isinstance(x, float) else x

mask = np.triu(np.ones_like(corr_table, dtype=bool))

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    corr_table,
    mask=mask,
    vmin=-1,
    vmax=1,
    cmap="RdBu",
    annot=True,
    fmt='.2f',
    annot_kws={"fontsize": 8, "color": "k"},
    cbar_kws={"format": '%.2f'}
)

for t in ax.texts:
    t.set_text(t.get_text().replace('.', ','))

cbar = ax.collections[0].colorbar
tick_labels = [label.get_text().replace('.', ',') for label in cbar.ax.get_yticklabels()]
cbar.ax.set_yticklabels(tick_labels)

plt.tight_layout()
plt.savefig('../figures/correlation_heatmap.png')
plt.show()

# %%
# Distribuição das Variáveis Numéricas em relação à variável Target
for var in num_features:

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(
        data=df_processed, 
        x=var, 
        hue=target, 
        kde=True, 
        bins=30, 
        palette={0:'#377eb8', 1: '#e41a1c'},  
        hue_order=[0, 1],
        ax=axes[0],
        legend=False
    )
    axes[0].set_title(f'Distribuição de {var} em relação à {target}', fontsize=14)
    axes[0].set_xlabel(var)
    axes[0].set_ylabel('Frequência')

    sns.boxplot(
        data=df_processed, 
        y=df_processed[target].astype('category'), 
        x=var, 
        palette={'0': '#377eb8', '1': '#e41a1c'},
        ax=axes[1]
    )
    axes[1].set_title(f'Análise de {var} em relação à target', fontsize=14)
    axes[1].set_xlabel(var)
    axes[1].set_ylabel('Target')
    axes[1].legend(loc='upper right', labels=['Ausência de Doença Cardíaca', 'Presença de Doença Cardíaca'])

    plt.tight_layout()
    plt.show()
# %%
# Boxplot para Variáveis Numéricas mais relevantes
fig, axes = plt.subplots(1, 2, figsize=(18, 8)) 

for i, var in enumerate(num_features[-2:]):
    ax = axes[i]
    sns.boxplot(
        data=df_processed, 
        y=df_processed[target].astype('category'), 
        x=var, 
        palette={'0': '#377eb8', '1': '#e41a1c'}, 
        ax=ax
    )
    ax.set_xlabel(var)
    ax.set_ylabel('Target')
    ax.legend(loc='upper right', labels=['Ausência de Doença Cardíaca', 'Presença de Doença Cardíaca'])

plt.tight_layout()
plt.savefig('../figures/boxplot_blood_pressure.png')
plt.show()
# %%
# Point plot e Distribuição de frequências para Variáveis Categóricas
for var in cat_features:
    fig, ax = plt.subplots(figsize=(10, 6))

    ordered_categories = sorted(df_processed[var].dropna().unique())

    sns.pointplot(data=df_processed, 
                  y=target,
                  x=var, 
                  order=ordered_categories,
                  ax=ax, 
                  capsize=.1,
                  color='#e41a1c')
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    sns.countplot(data=df_processed, 
                  x=var, 
                  order=ordered_categories,
                  hue=var, 
                  palette=['#377eb8'], 
                  legend=False,
                  alpha=0.3, 
                  ax=ax2)
    ax2.set_ylabel('Frequência')
    ax2.tick_params(axis='y')

# %%
selected_vars = ['cat_colesterol', 'cat_glicose']

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes = axes.flatten()

for i, var in enumerate(selected_vars):
    ax = axes[i]
    ordered_categories = sorted(df_processed[var].dropna().unique())

    sns.pointplot(
        data=df_processed,
        y=target,
        x=var,
        order=ordered_categories,
        ax=ax,
        capsize=0.1,
        color='#e41a1c'
    )
    ax.set_ylim(0, 1)
    ax.set_title(var.replace('_', ' ').title())
    ax.set_xlabel('')
    ax.set_ylabel('Probabilidade de Doença Cardíaca')

    ax2 = ax.twinx()
    sns.countplot(
        data=df_processed,
        x=var,
        order=ordered_categories,
        hue=var,
        palette=['#377eb8'],
        legend=False,
        alpha=0.3,
        ax=ax2
    )
    ax2.set_ylabel('Frequência')
    ax2.tick_params(axis='y')

plt.tight_layout()
plt.savefig('../figures/pointplot_cholesterol_glucose.png')
plt.show()

# %%
df_processed.to_csv('../data/processed/processed_cardio_data.csv', index=False)