# %%
import pandas as pd
import numpy as np

# %%
df_raw = pd.read_csv('../data/raw/raw_cardio_data.csv')	

df_raw.drop(columns=['age', 'bp_category_encoded'], inplace=True)

df_raw.rename(
    columns={
        'id': 'paciente_id', 
        'gender': 'cat_genero', 
        'height': 'vlr_altura', 
        'weight': 'vlr_peso', 
        'ap_hi': 'vlr_pressao_sistolica', 
        'ap_lo': 'vlr_pressao_diastolica', 
        'cholesterol': 'cat_colesterol',
        'gluc': 'cat_glicose', 
        'smoke': 'flag_fumante', 
        'alco': 'flag_consumo_alcool', 
        'active': 'flag_atividade_fisica', 
        'cardio': 'flag_doenca_cardiaca', 
        'age_years': 'nr_anos_idade', 
        'bmi': 'vlr_imc',
        'bp_category': 'cat_pressao_arterial'
    }, 
    inplace=True
)

# %%
qtd_duplicatas = sum(df_raw['paciente_id'].duplicated())
print(f'Quantidade de duplicatas: {qtd_duplicatas}')

# %%

cat_hipertensao_condicoes = [
    df_raw['cat_pressao_arterial'] == 'Hypertension Stage 1',
    df_raw['cat_pressao_arterial'] == 'Hypertension Stage 2',
    df_raw['cat_pressao_arterial'] == 'Normal',
    df_raw['cat_pressao_arterial'] == 'Elevated',
    df_raw['cat_pressao_arterial'] == 'Hypertensive Crisis'
]

novos_valores_cat_hipertensao = ['hipertensao_nivel_1', 'hipertensao_nivel_2', 'pressao_normal', 'pressao_elevada', 'crise_hipertensiva']

df_raw['cat_pressao_arterial'] = np.select(cat_hipertensao_condicoes, novos_valores_cat_hipertensao, default='Checar')

# %%

cat_genero_condicoes = [
    df_raw['cat_genero'] == 1,
    df_raw['cat_genero'] == 2
]
novos_valores_cat_genero = ['feminino', 'masculino']

df_raw['cat_genero'] = np.select(cat_genero_condicoes, novos_valores_cat_genero, default='Checar')

# %%

cat_colesterol_condicoes = [
    df_raw['cat_colesterol'] == 1,
    df_raw['cat_colesterol'] == 2,
    df_raw['cat_colesterol'] == 3
]
novos_valores_cat_colesterol = ['colesterol_normal', 'colesterol_acima_do_normal', 'colesterol_muito_acima_do_normal']

df_raw['cat_colesterol'] = np.select(cat_colesterol_condicoes, novos_valores_cat_colesterol, default='Checar')

# %%

cat_glicose_condicoes = [
    df_raw['cat_glicose'] == 1,
    df_raw['cat_glicose'] == 2,
    df_raw['cat_glicose'] == 3
]
novos_valores_cat_glicose = ['glicose_normal', 'glicose_acima_do_normal', 'glicose_muito_acima_do_normal']

df_raw['cat_glicose'] = np.select(cat_glicose_condicoes, novos_valores_cat_glicose, default='Checar')
# %%

df_raw = df_raw.astype(
    {'cat_genero': 'object',
     'cat_colesterol': 'object',
     'cat_glicose': 'object'}
)

df_processed = df_raw.reindex(
    columns=[
        'paciente_id',
        'nr_anos_idade', 
        'cat_genero', 
        'vlr_altura', 
        'vlr_peso', 
        'vlr_imc', 
        'vlr_pressao_sistolica', 
        'vlr_pressao_diastolica', 
        'cat_pressao_arterial',
        'cat_colesterol', 
        'cat_glicose', 
        'flag_fumante', 
        'flag_consumo_alcool', 
        'flag_atividade_fisica', 
        'flag_doenca_cardiaca'
    ]
)

# %%

df_processed.to_csv(
    '../data/processed/processed_cardio_data.csv', 
    index=False
)