# %% 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import shap

import config
config.set_plot_style()
# %%
model_series = pd.read_pickle("../data/predicted/model_series.pkl")
df_processed = pd.read_csv("../data/processed/processed_cardio_data.csv")
# %%
all_data = []

for model_name, model_info in model_series.items():

    for metric_name, value in model_info['train_metrics'].items():
        row = {
            'Tipo de Métrica': 'Treino',
            'Métrica': metric_name,
            model_name: value
        }
        all_data.append(row)

    for metric_name, value in model_info['test_metrics'].items():
        row = {
            'Tipo de Métrica': 'Teste',
            'Métrica': metric_name,
            model_name: value
        }
        all_data.append(row)

df_raw_metrics = pd.DataFrame(all_data)

df_metrics = df_raw_metrics.groupby(['Tipo de Métrica', 'Métrica']).sum().reset_index()
df_metrics

# %%
model_name = 'XGBoost'
best_model = model_series[model_name]

print(f'Modelo: {model_name}')

# %%
df_processed['pred_proba'] = best_model["model"].predict_proba(df_processed[best_model["features"]])[:, 1]

# %%
plt.figure(figsize=(10,6))
sns.kdeplot(
    data=df_processed, 
    x='pred_proba', 
    hue='flag_doenca_cardiaca', 
    fill=True, 
    common_norm=False,
    alpha=0.5,
    palette='Set1'
)
plt.xlabel('Probabilidade Predita')
plt.ylabel('Densidade')
plt.legend(title='Doença Cardíaca', labels=['Não', 'Sim'])
plt.grid(visible=False)  
plt.tight_layout()
plt.savefig('../figures/kde_plot_pred_proba.png')
plt.show()
# %%
precision, recall, thresholds = metrics.precision_recall_curve(df_processed['flag_doenca_cardiaca'], df_processed['pred_proba'])

plt.figure(figsize=(10,6))
plt.plot(thresholds, recall[:-1], label='Recall', color='#8ecae6')     
plt.plot(thresholds, precision[:-1], label='Precision', color='#ffb3b3') 
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(visible=False)  
plt.tight_layout()
plt.savefig('../figures/precision_recall_curve.png')
plt.show()

# %%
df_threshold = pd.DataFrame({
    'precision': precision[:-1],
    'recall': recall[:-1],
    'thresholds': thresholds
})

df_threshold_filtered = df_threshold[df_threshold["recall"] <= 0.81]
best_threshold_metrics = df_threshold_filtered.loc[df_threshold_filtered["recall"].idxmax()]

best_threshold_metrics

# %%
X_shap = best_model["model"].named_steps['preprocessor'].transform(df_processed[best_model["features"]]) 

underlying_model = best_model["model"].named_steps["classifier"] if hasattr(best_model["model"], "named_steps") else best_model["model"]

explainer = shap.TreeExplainer(underlying_model)
shap_values = explainer.shap_values(X_shap)

# %%
feature_names = [
    'Idade em anos', 'Altura', 'Peso',
    'IMC', 'Pressao Sistolica',
    'Pressao Diastolica', 'Masculino',
    'Pressao Elevada',
    'Hipertensao nivel 1',
    'Hipertensao nivel 2',
    'Colesterol acima do normal',
    'Colesterol muito acima do normal',
    'Glicose acima do normal',
    'Glicose muito acima do normal',
    'Fumante', 'Consumo de alcool',
    'Atividade fisica'
]
# %%
ax = shap.summary_plot(
    shap_values, 
    X_shap, 
    plot_type='bar', 
    feature_names=feature_names, 
    color='#377eb8',
    show=False
)
plt.xlabel('Importância Média do SHAP')
plt.tight_layout()
plt.savefig('../figures/shap_summary_bar.png')
plt.show()

# %%
shap.summary_plot(
    shap_values, 
    X_shap, 
    plot_type='dot', 
    feature_names=feature_names, 
    cmap='coolwarm',
    show=False
)
plt.xlabel('Valores de SHAP - Impacto no Modelo')
plt.tight_layout()
plt.savefig('../figures/shap_summary_dot.png')
plt.show()
